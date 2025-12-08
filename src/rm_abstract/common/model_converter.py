"""
Model Converter

Unified model format conversion utilities.
Handles ONNX conversion and other format transformations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """Configuration for model conversion"""

    # ONNX conversion options
    opset_version: int = 14
    batch_size: int = 1
    seq_length: int = 128
    dynamic_axes: bool = True

    # Input/Output names
    input_names: List[str] = field(default_factory=lambda: ["input_ids"])
    output_names: List[str] = field(default_factory=lambda: ["output"])

    # Cache settings
    cache_dir: Optional[str] = None
    use_cache: bool = True

    # Model-specific
    vocab_size: Optional[int] = None  # Auto-detect if None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "opset_version": self.opset_version,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "dynamic_axes": self.dynamic_axes,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "vocab_size": self.vocab_size,
        }


class ModelConverter:
    """
    Unified model converter for format transformations.

    Supports:
    - PyTorch to ONNX conversion
    - Cache management for converted models
    - Automatic dummy input generation

    Example:
        converter = ModelConverter(cache_dir="~/.cache/models")

        # Convert PyTorch model to ONNX
        onnx_path = converter.to_onnx(model, config=ConversionConfig(batch_size=1))

        # Get cache key for a model
        cache_key = converter.get_cache_key(model, config)
    """

    DEFAULT_CACHE_DIR = os.path.join(Path.home(), ".rm_abstract", "cache", "onnx")

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Directory for caching converted models
        """
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def to_onnx(
        self,
        model: Any,
        config: Optional[ConversionConfig] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert PyTorch model to ONNX format.

        Args:
            model: PyTorch model
            config: Conversion configuration
            output_path: Optional output path (uses cache if None)

        Returns:
            Path to ONNX file
        """
        import torch

        config = config or ConversionConfig()

        # Determine output path
        if output_path is None:
            cache_key = self.get_cache_key(model, config)
            output_path = os.path.join(self.cache_dir, f"{cache_key}.onnx")

        # Check cache
        if config.use_cache and os.path.exists(output_path):
            logger.debug(f"Using cached ONNX model: {output_path}")
            return output_path

        # Create dummy input
        dummy_input = self._create_dummy_input(model, config)

        # Prepare dynamic axes
        dynamic_axes = None
        if config.dynamic_axes:
            dynamic_axes = {
                name: {0: "batch_size", 1: "sequence"}
                for name in config.input_names + config.output_names
            }

        # Export to ONNX
        logger.info(f"Converting model to ONNX (opset={config.opset_version})...")

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=config.input_names,
            output_names=config.output_names,
            dynamic_axes=dynamic_axes,
            opset_version=config.opset_version,
        )

        logger.info(f"ONNX model saved to: {output_path}")
        return output_path

    def _create_dummy_input(
        self, model: Any, config: ConversionConfig
    ) -> Any:
        """
        Create dummy input for ONNX export.

        Args:
            model: PyTorch model
            config: Conversion configuration

        Returns:
            Dummy input tensor(s)
        """
        import torch

        # Determine vocab size
        vocab_size = config.vocab_size
        if vocab_size is None:
            vocab_size = self._get_vocab_size(model)

        # Create dummy input IDs
        dummy_input = torch.randint(
            0, vocab_size, (config.batch_size, config.seq_length)
        )

        return dummy_input

    def _get_vocab_size(self, model: Any) -> int:
        """
        Get vocabulary size from model.

        Args:
            model: PyTorch model

        Returns:
            Vocabulary size (default 32000 if not found)
        """
        if hasattr(model, "config"):
            vocab_size = getattr(model.config, "vocab_size", None)
            if vocab_size is not None:
                return vocab_size

        # Try to get from embedding layer
        if hasattr(model, "embed_tokens"):
            return model.embed_tokens.weight.shape[0]
        if hasattr(model, "embeddings"):
            if hasattr(model.embeddings, "word_embeddings"):
                return model.embeddings.word_embeddings.weight.shape[0]

        # Default fallback
        logger.warning("Could not detect vocab_size, using default 32000")
        return 32000

    def get_cache_key(
        self, model: Any, config: Optional[ConversionConfig] = None
    ) -> str:
        """
        Generate cache key for a model.

        Args:
            model: Model object
            config: Conversion configuration

        Returns:
            Cache key string (MD5 hash)
        """
        # Get model identifier
        model_name = self._get_model_name(model)

        # Include config in key
        config_str = ""
        if config is not None:
            config_str = str(sorted(config.to_dict().items()))

        key_string = f"{model_name}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def _get_model_name(self, model: Any) -> str:
        """
        Get model name/identifier.

        Args:
            model: Model object

        Returns:
            Model name string
        """
        if isinstance(model, str):
            return model

        if hasattr(model, "name_or_path"):
            return model.name_or_path

        if hasattr(model, "config"):
            name = getattr(model.config, "name_or_path", None)
            if name:
                return name
            # Try _name_or_path
            name = getattr(model.config, "_name_or_path", None)
            if name:
                return name

        return type(model).__name__

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Extract model information.

        Args:
            model: Model object

        Returns:
            Dictionary with model info
        """
        info = {
            "name": self._get_model_name(model),
            "type": type(model).__name__,
            "vocab_size": self._get_vocab_size(model),
        }

        if hasattr(model, "config"):
            config = model.config
            info.update(
                {
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                }
            )

        return info

    def clear_cache(self, model: Optional[Any] = None) -> int:
        """
        Clear cached models.

        Args:
            model: If provided, clear only this model's cache

        Returns:
            Number of files deleted
        """
        deleted = 0

        if model is not None:
            # Delete specific model cache
            cache_key = self.get_cache_key(model)
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.onnx")
            if os.path.exists(cache_path):
                os.remove(cache_path)
                deleted = 1
        else:
            # Delete all cached models
            for f in os.listdir(self.cache_dir):
                if f.endswith(".onnx"):
                    os.remove(os.path.join(self.cache_dir, f))
                    deleted += 1

        logger.info(f"Cleared {deleted} cached model(s)")
        return deleted

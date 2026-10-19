"""
Transformers Compatibility Layer

Provides drop-in replacement for transformers that routes to appropriate backend
based on RM_DEVICE environment variable or explicit configuration.

Usage:
    # Original code (no modification needed with auto-hook)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Or explicit import
    from rm_abstract.compat import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
"""

import logging
import os
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Get device from environment or auto-detect."""
    return os.environ.get("RM_DEVICE", "auto")


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device."""
    if device != "auto":
        return device.split(":")[0]
    return _detect_device()


def _detect_device() -> str:
    """Detect the best available device on this host (NPU > GPU > CPU)."""
    try:
        import rebel  # noqa: F401

        return "rbln"
    except ImportError:
        pass

    try:
        import furiosa_sdk  # noqa: F401

        return "furiosa"
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            return "nvidia"
    except ImportError:
        pass

    return "cpu"


class AutoModelForCausalLM:
    """
    Transformers-compatible AutoModelForCausalLM.

    Automatically routes to the appropriate backend based on device selection.

    Example:
        # Environment variable: RM_DEVICE=rbln
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        # Or explicit device
        model = AutoModelForCausalLM.from_pretrained("gpt2", device="rbln")
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
        device_map: Optional[Union[str, Dict]] = None,
        **kwargs,
    ) -> Any:
        """
        Load a pretrained model with automatic backend selection.

        Args:
            model_name_or_path: Model identifier or path
            device: Device selection (auto, cpu, nvidia, rbln, furiosa)
            trust_remote_code: Trust remote code
            torch_dtype: PyTorch dtype
            device_map: Device mapping (auto, balanced, etc.)
            **kwargs: Additional arguments

        Returns:
            Loaded model instance
        """
        device = device or _get_device()
        resolved_device = _resolve_device(device)

        logger.info(f"Loading model {model_name_or_path} on device: {resolved_device}")

        if resolved_device == "rbln":
            return cls._load_rbln(model_name_or_path, trust_remote_code, **kwargs)
        elif resolved_device == "furiosa":
            return cls._load_furiosa(model_name_or_path, trust_remote_code, **kwargs)
        elif resolved_device in ("nvidia", "gpu", "cuda"):
            return cls._load_gpu(
                model_name_or_path, trust_remote_code, torch_dtype, device_map, **kwargs
            )
        else:
            return cls._load_cpu(
                model_name_or_path, trust_remote_code, torch_dtype, **kwargs
            )

    @classmethod
    def _load_rbln(cls, model_name: str, trust_remote_code: bool, **kwargs) -> Any:
        """Load model using Rebellions RBLN backend."""
        try:
            from optimum.rbln import RBLNAutoModelForCausalLM

            model = RBLNAutoModelForCausalLM.from_pretrained(
                model_name, export=True, trust_remote_code=trust_remote_code, **kwargs
            )
            logger.info(f"Loaded model with optimum-rbln: {model_name}")
            return model

        except ImportError:
            logger.warning("optimum-rbln not available, falling back to CPU")
            return cls._load_cpu(model_name, trust_remote_code, **kwargs)

    @classmethod
    def _load_furiosa(cls, model_name: str, trust_remote_code: bool, **kwargs) -> Any:
        """Load model using FuriosaAI backend."""
        try:
            from furiosa_llm import LLM as FuriosaLLM

            model = FuriosaLLM.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, **kwargs
            )
            logger.info(f"Loaded model with FuriosaAI: {model_name}")
            return model

        except ImportError:
            logger.warning("FuriosaAI backend not available, falling back to CPU")
            return cls._load_cpu(model_name, trust_remote_code, **kwargs)

    @classmethod
    def _get_hf_automodel(cls):
        """Get original HuggingFace AutoModelForCausalLM without hooks."""
        import importlib
        import sys

        from rm_abstract.auto import disable_hooks, enable_hooks

        disable_hooks()

        try:
            if "transformers" in sys.modules:
                del sys.modules["transformers"]
            transformers = importlib.import_module("transformers")
            return transformers.AutoModelForCausalLM
        finally:
            enable_hooks()

    @classmethod
    def _load_gpu(
        cls,
        model_name: str,
        trust_remote_code: bool,
        torch_dtype: Any,
        device_map: Optional[Union[str, Dict]],
        **kwargs,
    ) -> Any:
        """Load model on GPU using transformers."""
        import torch

        HFAutoModel = cls._get_hf_automodel()

        if torch_dtype is None:
            torch_dtype = torch.float16

        if device_map is None:
            device_map = "auto"

        model = HFAutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        logger.info(f"Loaded model on GPU: {model_name}")
        return model

    @classmethod
    def _load_cpu(
        cls, model_name: str, trust_remote_code: bool, torch_dtype: Any = None, **kwargs
    ) -> Any:
        """Load model on CPU using transformers."""
        import torch

        HFAutoModel = cls._get_hf_automodel()

        if torch_dtype is None:
            torch_dtype = torch.float32

        model = HFAutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="cpu",
            **kwargs,
        )
        logger.info(f"Loaded model on CPU: {model_name}")
        return model


class AutoTokenizer:
    """
    Transformers-compatible AutoTokenizer.

    This is a simple pass-through to HuggingFace's AutoTokenizer since
    tokenizers are device-agnostic.
    """

    @classmethod
    def _get_hf_autotokenizer(cls):
        """Get original HuggingFace AutoTokenizer without hooks."""
        import importlib
        import sys

        from rm_abstract.auto import disable_hooks, enable_hooks

        disable_hooks()

        try:
            if "transformers" in sys.modules:
                del sys.modules["transformers"]
            transformers = importlib.import_module("transformers")
            return transformers.AutoTokenizer
        finally:
            enable_hooks()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> Any:
        """
        Load a pretrained tokenizer.

        Args:
            model_name_or_path: Model identifier or path
            **kwargs: Additional arguments

        Returns:
            Tokenizer instance
        """
        HFAutoTokenizer = cls._get_hf_autotokenizer()
        return HFAutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

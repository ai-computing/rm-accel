"""
FuriosaAI RNGD NPU Resource

NPU resource implementation for FuriosaAI RNGD chips.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import os
import logging

from ..base import NPUResource, ResourceType, ResourceInfo, ResourceCapability
from ...common.input_adapter import InputAdapter
from ...common.output_adapter import OutputAdapter
from ...common.model_converter import ModelConverter, ConversionConfig

logger = logging.getLogger(__name__)


class FuriosaRNGD(NPUResource):
    """
    FuriosaAI RNGD NPU Resource

    Performs LLM inference on RNGD NPU via Furiosa SDK.

    Example:
        npu = FuriosaRNGD(device_id=0)
        npu.initialize()

        model = npu.prepare_model(pytorch_model)
        output = npu.execute(model, inputs)

        npu.cleanup()
    """

    RESOURCE_TYPE = ResourceType.NPU
    VENDOR = "furiosa"
    NAME = "FuriosaAI RNGD"

    # Default compilation options
    DEFAULT_TARGET = "rngd"
    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(device_id, cache_dir, compile_options, **kwargs)
        self._runner = None
        self._input_adapter = InputAdapter()
        self._output_adapter = OutputAdapter()
        self._model_converter = ModelConverter(cache_dir)

        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = os.path.join(
                Path.home(), ".rm_abstract", "cache", "furiosa"
            )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def compiled_model_extension(self) -> str:
        return "enf"

    @staticmethod
    def is_available() -> bool:
        """Check Furiosa SDK and NPU availability"""
        try:
            from furiosa import runtime

            devices = runtime.list_devices()
            return len(devices) > 0
        except ImportError:
            logger.debug("Furiosa SDK not installed")
            return False
        except Exception as e:
            logger.debug(f"Furiosa NPU not available: {e}")
            return False

    def initialize(self) -> None:
        """Initialize Furiosa resource"""
        if self._initialized:
            return

        try:
            from furiosa import runtime

            devices = runtime.list_devices()
            if self.device_id >= len(devices):
                raise RuntimeError(
                    f"Furiosa device {self.device_id} not found. "
                    f"Available: {len(devices)}"
                )

            self._initialized = True
            logger.info(f"FuriosaAI RNGD initialized on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize FuriosaAI RNGD: {e}")
            raise

    def prepare_model(
        self, model: Any, config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Prepare model for Furiosa NPU.

        Pipeline: PyTorch → ONNX → ENF (Furiosa compiled format)

        Args:
            model: PyTorch model or model path
            config: Model configuration

        Returns:
            Compiled Furiosa model (ENF binary)
        """
        if not self._initialized:
            self.initialize()

        config = config or {}
        compile_config = {**self.compile_options, **config}

        # Generate cache key
        cache_key = self._model_converter.get_cache_key(model, ConversionConfig(**{
            k: v for k, v in compile_config.items()
            if k in ["batch_size", "seq_length", "opset_version"]
        }))
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.{self.compiled_model_extension}")

        # Check cache
        if os.path.exists(cache_path):
            logger.info(f"Loading compiled model from cache: {cache_path}")
            return self.load_compiled_model(cache_path)

        # Convert to ONNX
        logger.info("Converting model to ONNX...")
        conversion_config = ConversionConfig(
            batch_size=compile_config.get("batch_size", self.DEFAULT_BATCH_SIZE),
            seq_length=compile_config.get("seq_length", 128),
            opset_version=compile_config.get("opset_version", 14),
            cache_dir=self.cache_dir,
        )
        onnx_path = self._model_converter.to_onnx(model, conversion_config)

        # Compile for Furiosa NPU
        logger.info("Compiling model for FuriosaAI RNGD...")
        compiled_model = self.compile_model(onnx_path, **compile_config)

        # Save to cache
        logger.info(f"Saving compiled model to cache: {cache_path}")
        self.save_compiled_model(compiled_model, cache_path)

        return compiled_model

    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        Compile model for FuriosaAI RNGD NPU.

        Args:
            model: ONNX model path
            **kwargs: Compilation options

        Returns:
            Compiled Furiosa model (ENF format)
        """
        from furiosa import compiler

        batch_size = kwargs.get("batch_size", self.DEFAULT_BATCH_SIZE)
        target = kwargs.get("target", self.DEFAULT_TARGET)

        logger.info(f"Compiling for RNGD (target={target}, batch_size={batch_size})")

        if isinstance(model, str):
            compiled = compiler.compile(
                model,
                target=target,
                batch_size=batch_size,
            )
        else:
            raise ValueError("Furiosa compiler requires ONNX model path")

        return compiled

    def load_compiled_model(self, path: str) -> Any:
        """Load compiled Furiosa model"""
        logger.info(f"Loading compiled model from: {path}")
        with open(path, "rb") as f:
            return f.read()

    def save_compiled_model(self, model: Any, path: str) -> None:
        """Save compiled Furiosa model"""
        with open(path, "wb") as f:
            f.write(model)
        logger.info(f"Compiled model saved to: {path}")

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on FuriosaAI RNGD NPU.

        Args:
            model: Compiled Furiosa model (ENF binary)
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        if proxy_method == "generate":
            return self._execute_generate(model, inputs, original_model, **kwargs)
        else:
            return self._execute_forward(model, inputs, **kwargs)

    def _execute_forward(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute single forward pass on NPU"""
        from furiosa import runtime
        import numpy as np

        # Convert inputs to numpy
        if hasattr(inputs, "numpy"):
            inputs = inputs.numpy()
        elif isinstance(inputs, dict):
            inputs = {
                k: v.numpy() if hasattr(v, "numpy") else v
                for k, v in inputs.items()
            }

        # Create runner if needed
        if self._runner is None:
            self._runner = runtime.create_runner(model, device=f"npu{self.device_id}")

        # Execute
        if isinstance(inputs, dict):
            outputs = self._runner.run(**inputs)
        else:
            outputs = self._runner.run(inputs)

        return outputs

    def _execute_generate(
        self, model: Any, inputs: Any, original_model: Any, **kwargs
    ) -> Any:
        """Execute text generation on NPU"""
        import torch

        max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 128))

        # Get input tensor
        if hasattr(inputs, "input_ids"):
            input_ids = inputs.input_ids
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)

        # Greedy generation loop
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self._execute_forward(model, generated, **kwargs)

            # Get next token
            if hasattr(outputs, "logits"):
                next_token_logits = outputs.logits[:, -1, :]
            elif isinstance(outputs, dict) and "logits" in outputs:
                next_token_logits = outputs["logits"][:, -1, :]
            else:
                if hasattr(outputs, "shape") and len(outputs.shape) >= 2:
                    next_token_logits = outputs[:, -1, :]
                else:
                    next_token_logits = outputs

            if not isinstance(next_token_logits, torch.Tensor):
                next_token_logits = torch.tensor(next_token_logits)

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if original_model is not None and hasattr(original_model, "config"):
                eos_token_id = getattr(original_model.config, "eos_token_id", None)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    def get_info(self) -> ResourceInfo:
        """Return FuriosaAI NPU information"""
        try:
            from furiosa import runtime

            devices = runtime.list_devices()
            if self.device_id < len(devices):
                device = devices[self.device_id]
                return ResourceInfo(
                    resource_type=ResourceType.NPU,
                    device_id=self.device_id,
                    name=self.NAME,
                    vendor=self.VENDOR,
                    extra={"device_info": str(device)},
                )
        except Exception:
            pass

        return ResourceInfo(
            resource_type=ResourceType.NPU,
            device_id=self.device_id,
            name=self.NAME,
            vendor=self.VENDOR,
        )

    def get_capability(self) -> ResourceCapability:
        """Return Furiosa capabilities"""
        return ResourceCapability(
            supports_fp16=True,
            supports_int8=True,
            supports_int4=True,
        )

    def cleanup(self) -> None:
        """Cleanup Furiosa resources"""
        if self._runner is not None:
            try:
                self._runner.close()
            except Exception:
                pass
            self._runner = None
        super().cleanup()

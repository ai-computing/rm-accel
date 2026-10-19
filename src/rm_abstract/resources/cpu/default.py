"""
CPU Resource

Fallback resource using PyTorch CPU inference.
"""

import logging
import os
import platform
from typing import Any, Dict, Optional

from ...common.input_adapter import InputAdapter
from ..base import Resource, ResourceCapability, ResourceInfo, ResourceType

logger = logging.getLogger(__name__)


class CPUResource(Resource):
    """
    CPU Resource

    Fallback resource for CPU-only inference using PyTorch.

    Example:
        cpu = CPUResource()
        cpu.initialize()

        model = cpu.prepare_model(pytorch_model)
        output = cpu.execute(model, inputs)

        cpu.cleanup()
    """

    RESOURCE_TYPE = ResourceType.CPU
    VENDOR = "generic"
    NAME = "CPU"

    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)
        self._input_adapter = InputAdapter()

    @staticmethod
    def is_available() -> bool:
        """CPU is always available if PyTorch is installed"""
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize CPU resource"""
        if self._initialized:
            return

        self._initialized = True
        logger.info("CPU resource initialized")

    def prepare_model(self, model: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model for CPU inference.

        Args:
            model: PyTorch model
            config: Model configuration

        Returns:
            Model on CPU in eval mode
        """

        if hasattr(model, "to"):
            model = model.to("cpu")

        if hasattr(model, "eval"):
            model.eval()

        logger.info("Model prepared for CPU inference")
        return model

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on CPU.

        Args:
            model: PyTorch model
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        import torch

        proxy_method = kwargs.pop("_proxy_method", None)
        kwargs.pop("original_model", None)

        with torch.no_grad():
            # Prepare inputs
            exec_inputs, exec_kwargs = self._prepare_execution(inputs, kwargs)

            # Route based on proxy method
            if proxy_method == "generate" and hasattr(model, "generate"):
                if exec_inputs is None:
                    return model.generate(**exec_kwargs)
                else:
                    return model.generate(exec_inputs, **exec_kwargs)

            elif proxy_method in ("forward", "__call__"):
                if exec_inputs is None:
                    return model(**exec_kwargs)
                else:
                    return model(exec_inputs, **exec_kwargs)

            else:
                # Default: use generate if available, otherwise forward
                if hasattr(model, "generate"):
                    if exec_inputs is None:
                        return model.generate(**exec_kwargs)
                    else:
                        return model.generate(exec_inputs, **exec_kwargs)
                else:
                    if exec_inputs is None:
                        return model(**exec_kwargs)
                    else:
                        return model(exec_inputs, **exec_kwargs)

    def _prepare_execution(self, inputs: Any, kwargs: Dict[str, Any]) -> tuple:
        """Prepare inputs for execution"""
        import torch

        if inputs is None:
            # All inputs in kwargs
            exec_kwargs = {
                k: v.to("cpu") if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
            return None, exec_kwargs

        elif isinstance(inputs, dict):
            # Dict inputs - merge with kwargs
            exec_kwargs = {**kwargs}
            for k, v in inputs.items():
                exec_kwargs[k] = v.to("cpu") if isinstance(v, torch.Tensor) else v
            return None, exec_kwargs

        elif isinstance(inputs, torch.Tensor):
            return inputs.to("cpu"), kwargs

        else:
            return inputs, kwargs

    def get_info(self) -> ResourceInfo:
        """Return CPU information"""
        return ResourceInfo(
            resource_type=ResourceType.CPU,
            device_id=0,
            name=platform.processor() or "CPU",
            vendor=platform.system(),
            extra={
                "cpu_count": os.cpu_count(),
                "platform": platform.platform(),
            },
        )

    def get_capability(self) -> ResourceCapability:
        """Return CPU capabilities"""
        return ResourceCapability(
            supports_fp16=False,  # CPU FP16 is slow
            supports_fp32=True,
            supports_int8=True,
        )

    def cleanup(self) -> None:
        """Cleanup CPU resources"""
        super().cleanup()

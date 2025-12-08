"""
RM Abstract Layer - Heterogeneous AI Accelerator Unified Compatibility Library

A unified interface for running LLM inference across different hardware
(GPU, NPU, CPU) and serving engines (vLLM, Triton, TorchServe).

Quick Start:
    from rm_abstract import LLM, SamplingParams

    llm = LLM(model="gpt2", device="auto")
    outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=50))
    print(outputs[0].outputs[0].text)

Architecture:
- llm.py: Main LLM proxy class (recommended entry point)
- common/: Shared utilities (InputAdapter, OutputAdapter, ModelConverter, HardwareDetector)
- resources/: Hardware-specific implementations (GPU, NPU, CPU)
- engines/: Serving engines (vLLM, Triton, TorchServe)
- registry/: Centralized plugin registration system
"""

from typing import Optional, Dict, Any
import os

from .core.controller import DeviceFlowController
from .core.config import Config
from .exceptions import (
    RMAbstractError,
    NotInitializedError,
    BackendNotAvailableError,
    InvalidDeviceError,
)

# Main LLM interface (recommended)
from .llm import LLM, SamplingParams, RequestOutput, CompletionOutput

# Modular components
from .registry import PluginRegistry, PluginType, get_registry, discover_all
from .common import InputAdapter, OutputAdapter, ModelConverter, HardwareDetector

__version__ = "0.3.0"
__all__ = [
    # Main LLM interface (recommended)
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    # Core functions
    "init",
    "is_initialized",
    "switch_device",
    "get_device_info",
    "get_controller",
    "get_available_resources",
    # Modular components
    "PluginRegistry",
    "PluginType",
    "get_registry",
    "InputAdapter",
    "OutputAdapter",
    "ModelConverter",
    "HardwareDetector",
    # Exceptions
    "RMAbstractError",
    "NotInitializedError",
    "BackendNotAvailableError",
    "InvalidDeviceError",
]

# Global controller instance
_global_controller: Optional[DeviceFlowController] = None
_initialized: bool = False


def init(
    device: str = "auto",
    cache_dir: Optional[str] = None,
    compile_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> DeviceFlowController:
    """
    Initialize RM Abstract Layer

    Args:
        device: Device specification
            - "auto": Auto-select (NPU > GPU > CPU priority)
            - "nvidia" or "gpu": NVIDIA GPU
            - "rbln": Rebellions ATOM NPU
            - "furiosa": FuriosaAI RNGD NPU
            - "cpu": CPU fallback
        cache_dir: NPU compilation cache directory (default: ~/.rm_abstract/cache)
        compile_options: NPU compilation options
        verbose: Whether to print initialization info

    Returns:
        DeviceFlowController instance

    Example:
        >>> import rm_abstract
        >>> rm_abstract.init(device="auto")
    """
    global _global_controller, _initialized

    device = os.environ.get("RM_DEVICE", device)
    cache_dir = os.environ.get("RM_CACHE_DIR", cache_dir)

    config = Config(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options or {},
        verbose=verbose,
    )

    _global_controller = DeviceFlowController(config)
    _initialized = True

    if verbose:
        print(f"[RM Abstract] Initialized with device: {_global_controller.device_name}")

    return _global_controller


def switch_device(device: str) -> None:
    """Switch device at runtime"""
    if _global_controller is None:
        raise NotInitializedError()
    _global_controller.switch_device(device)


def get_device_info() -> Dict[str, Any]:
    """Return current device information"""
    if _global_controller is None:
        raise NotInitializedError()
    return _global_controller.get_device_info()


def get_controller() -> Optional[DeviceFlowController]:
    """Return current global controller"""
    return _global_controller


def get_available_resources() -> Dict[str, bool]:
    """
    Get available resources from the plugin registry.

    Returns:
        Dictionary of {resource_name: is_available}
    """
    registry = get_registry()
    if not registry.list_plugins():
        discover_all(registry)

    result = {}
    for plugin_info in registry.list_plugins(PluginType.RESOURCE):
        result[plugin_info.name] = plugin_info.is_available()

    return result


def is_initialized() -> bool:
    """Check if RM Abstract Layer is initialized"""
    return _initialized

"""
RM Abstract Layer - Heterogeneous AI Accelerator Unified Compatibility Library

An abstraction layer that enables existing GPU inference scripts to run on NPU/GPU
without any code modification.
"""

from typing import Optional, Dict, Any, List
import os

from .core.controller import DeviceFlowController
from .core.config import Config
from .core.resource_manager import ResourceManager
from .core.plugin import PluginMetadata
from .system_info import get_system_info, print_system_info, get_quick_status
from .system_validator import validate_system, print_validation_report, get_working_components
from .exceptions import (
    RMAbstractError,
    NotInitializedError,
    BackendNotAvailableError,
    InvalidDeviceError,
)

__version__ = "0.1.0"
__all__ = [
    "init",
    "switch_device",
    "get_device_info",
    "get_controller",
    "get_available_backends",
    "get_resource_manager",
    "list_plugins",
    # System info (static check)
    "get_system_info",
    "print_system_info",
    "get_quick_status",
    # System validation (actual tests)
    "validate_system",
    "print_validation_report",
    "get_working_components",
    # Exceptions
    "RMAbstractError",
    "NotInitializedError",
    "BackendNotAvailableError",
    "InvalidDeviceError",
]

# Global controller instance (legacy)
_global_controller: Optional[DeviceFlowController] = None

# Global resource manager instance (new plugin system)
_global_resource_manager: Optional[ResourceManager] = None


def init(
    device: str = "auto",
    cache_dir: Optional[str] = None,
    compile_options: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    use_plugin_system: bool = False,
) -> DeviceFlowController:
    """
    Initialize RM Abstract Layer

    Args:
        device: Device specification
            - "auto": Auto-select (NPU > GPU > CPU priority)
            - "gpu:0", "gpu:1": Specific GPU
            - "rbln:0": Rebellions ATOM NPU
            - "furiosa:0": FuriosaAI RNGD NPU
            - "cpu": CPU
        cache_dir: NPU compilation cache directory (default: ~/.rm_abstract/cache)
        compile_options: NPU compilation options
        verbose: Whether to print compilation progress
        use_plugin_system: Use new plugin-based system (experimental)

    Returns:
        DeviceFlowController instance

    Example:
        >>> import rm_abstract
        >>> rm_abstract.init(device="rbln:0")
        >>> # Use existing code as-is
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    """
    global _global_controller, _global_resource_manager

    # Load settings from environment variables
    device = os.environ.get("RM_DEVICE", device)
    cache_dir = os.environ.get("RM_CACHE_DIR", cache_dir)
    use_plugin_system = os.environ.get("RM_USE_PLUGINS", str(use_plugin_system)).lower() in (
        "true",
        "1",
        "yes",
    )

    # Create Config
    config = Config(
        device=device,
        cache_dir=cache_dir,
        compile_options=compile_options or {},
        verbose=verbose,
    )

    if use_plugin_system:
        # Use new plugin-based system
        from .backends.auto_register import auto_register_backends

        auto_register_backends()
        _global_resource_manager = ResourceManager(config)
        _global_resource_manager.initialize(auto_discover=True)

        if verbose:
            print(
                f"[RM Abstract] Initialized with plugin: {_global_resource_manager.device_name}"
            )
    else:
        # Use legacy Backend system
        _register_backends()
        _global_controller = DeviceFlowController(config)
        _global_controller.activate_hooks()

        if verbose:
            print(f"[RM Abstract] Initialized with device: {_global_controller.device_name}")

    return _global_controller


def switch_device(device: str) -> None:
    """
    Switch device at runtime

    Args:
        device: New device specification
    """
    global _global_controller

    if _global_controller is None:
        raise NotInitializedError()

    _global_controller.switch_device(device)


def get_device_info() -> Dict[str, Any]:
    """
    Return current device information

    Returns:
        Device information dictionary
    """
    global _global_controller

    if _global_controller is None:
        raise NotInitializedError()

    return _global_controller.get_device_info()


def get_controller() -> Optional[DeviceFlowController]:
    """
    Return current global controller

    Returns:
        DeviceFlowController instance or None
    """
    return _global_controller


def get_available_backends() -> Dict[str, bool]:
    """
    Return available backends

    Returns:
        Dictionary of {backend_name: is_available}
    """
    # Register backends first
    _register_backends()
    return DeviceFlowController.get_available_backends()


def get_resource_manager() -> Optional[ResourceManager]:
    """
    Get global resource manager instance (new plugin system)

    Returns:
        ResourceManager instance or None
    """
    return _global_resource_manager


def list_plugins(available_only: bool = True) -> List[Dict[str, Any]]:
    """
    List all available plugins

    Args:
        available_only: Only show available plugins

    Returns:
        List of plugin information dictionaries
    """
    if _global_resource_manager is None:
        # Initialize plugin registry
        from .backends.auto_register import auto_register_backends

        auto_register_backends()

    from .backends.auto_register import list_registered_plugins

    return list_registered_plugins()


def _register_backends() -> None:
    """Register all available backends (legacy system)"""
    # CPU backend (always available if torch is installed)
    try:
        from .backends.cpu.cpu_backend import CPUBackend

        DeviceFlowController.register_backend("cpu", CPUBackend)
    except ImportError:
        pass

    # GPU backend (vLLM)
    try:
        from .backends.gpu.vllm_backend import VLLMBackend

        DeviceFlowController.register_backend("gpu", VLLMBackend)
    except ImportError:
        pass

    # Rebellions NPU backend
    try:
        from .backends.npu.plugins.rebellions import RBLNBackend

        DeviceFlowController.register_backend("rbln", RBLNBackend)
    except ImportError:
        pass

    # FuriosaAI NPU backend
    try:
        from .backends.npu.plugins.furiosa import FuriosaBackend

        DeviceFlowController.register_backend("furiosa", FuriosaBackend)
    except ImportError:
        pass

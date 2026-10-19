"""
Plugin Discovery

Auto-discovery and registration of available plugins.
"""

import logging
from typing import List, Optional

from .plugin_registry import (
    PluginInfo,
    PluginRegistry,
    PluginType,
    get_registry,
)

logger = logging.getLogger(__name__)


def discover_resources(registry: Optional[PluginRegistry] = None) -> List[PluginInfo]:
    """
    Discover and register available resource plugins.

    Args:
        registry: Plugin registry (uses global if None)

    Returns:
        List of discovered PluginInfo
    """
    registry = registry or get_registry()
    discovered = []

    # GPU Resources
    discovered.extend(_discover_gpu_resources(registry))

    # NPU Resources
    discovered.extend(_discover_npu_resources(registry))

    # CPU Resource (always available)
    discovered.extend(_discover_cpu_resources(registry))

    return discovered


def _discover_gpu_resources(registry: PluginRegistry) -> List[PluginInfo]:
    """Discover GPU resources"""
    discovered = []

    # NVIDIA GPU
    try:
        from ..resources.gpu.nvidia import NvidiaGPU

        plugin_info = PluginInfo(
            name="nvidia",
            plugin_type=PluginType.RESOURCE,
            cls=NvidiaGPU,
            vendor="nvidia",
            description="NVIDIA GPU with vLLM",
            priority=50,
            aliases=["gpu", "cuda", "vllm_gpu"],
            requirements=["torch", "vllm"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: NVIDIA GPU resource")
    except ImportError as e:
        logger.debug(f"NVIDIA GPU resource not available: {e}")

    return discovered


def _discover_npu_resources(registry: PluginRegistry) -> List[PluginInfo]:
    """Discover NPU resources"""
    discovered = []

    # Rebellions ATOM NPU
    try:
        from ..resources.npu.rebellions import RebellionsATOM

        plugin_info = PluginInfo(
            name="rbln",
            plugin_type=PluginType.RESOURCE,
            cls=RebellionsATOM,
            vendor="rebellions",
            description="Rebellions ATOM NPU",
            priority=100,  # NPU preferred over GPU
            aliases=["rebellions", "atom", "npu_rbln"],
            requirements=["rebel", "vllm"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: Rebellions ATOM NPU resource")
    except ImportError as e:
        logger.debug(f"Rebellions ATOM resource not available: {e}")

    # FuriosaAI RNGD NPU
    try:
        from ..resources.npu.furiosa import FuriosaRNGD

        plugin_info = PluginInfo(
            name="furiosa",
            plugin_type=PluginType.RESOURCE,
            cls=FuriosaRNGD,
            vendor="furiosa",
            description="FuriosaAI RNGD NPU",
            priority=100,  # NPU preferred over GPU
            aliases=["rngd", "npu_furiosa"],
            requirements=["furiosa"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: FuriosaAI RNGD NPU resource")
    except ImportError as e:
        logger.debug(f"FuriosaAI RNGD resource not available: {e}")

    return discovered


def _discover_cpu_resources(registry: PluginRegistry) -> List[PluginInfo]:
    """Discover CPU resources"""
    discovered = []

    try:
        from ..resources.cpu.default import CPUResource

        plugin_info = PluginInfo(
            name="cpu",
            plugin_type=PluginType.RESOURCE,
            cls=CPUResource,
            vendor="generic",
            description="CPU fallback resource",
            priority=0,  # Lowest priority (fallback)
            aliases=["pytorch_cpu"],
            requirements=["torch"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: CPU resource")
    except ImportError as e:
        logger.debug(f"CPU resource not available: {e}")

    return discovered


def discover_engines(registry: Optional[PluginRegistry] = None) -> List[PluginInfo]:
    """
    Discover and register available serving engine plugins.

    Args:
        registry: Plugin registry (uses global if None)

    Returns:
        List of discovered PluginInfo
    """
    registry = registry or get_registry()
    discovered = []

    # vLLM Engine
    try:
        from ..engines.vllm import VLLMEngine

        plugin_info = PluginInfo(
            name="vllm",
            plugin_type=PluginType.ENGINE,
            cls=VLLMEngine,
            vendor="vllm",
            description="vLLM serving engine",
            priority=100,  # Preferred engine
            aliases=["vllm_engine"],
            requirements=["vllm"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: vLLM engine")
    except ImportError as e:
        logger.debug(f"vLLM engine not available: {e}")

    # Triton Engine
    try:
        from ..engines.triton import TritonEngine

        plugin_info = PluginInfo(
            name="triton",
            plugin_type=PluginType.ENGINE,
            cls=TritonEngine,
            vendor="nvidia",
            description="Triton Inference Server",
            priority=80,
            aliases=["triton_engine", "tritonserver"],
            requirements=["docker"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: Triton engine")
    except ImportError as e:
        logger.debug(f"Triton engine not available: {e}")

    # TorchServe Engine
    try:
        from ..engines.torchserve import TorchServeEngine

        plugin_info = PluginInfo(
            name="torchserve",
            plugin_type=PluginType.ENGINE,
            cls=TorchServeEngine,
            vendor="pytorch",
            description="TorchServe serving engine",
            priority=50,
            aliases=["ts", "torch_serve"],
            requirements=["torchserve"],
        )
        registry.register(plugin_info)
        discovered.append(plugin_info)
        logger.debug("Discovered: TorchServe engine")
    except ImportError as e:
        logger.debug(f"TorchServe engine not available: {e}")

    return discovered


def discover_all(registry: Optional[PluginRegistry] = None) -> List[PluginInfo]:
    """
    Discover and register all available plugins.

    Args:
        registry: Plugin registry (uses global if None)

    Returns:
        List of all discovered PluginInfo
    """
    discovered = []
    discovered.extend(discover_resources(registry))
    discovered.extend(discover_engines(registry))
    return discovered


def auto_register() -> None:
    """
    Auto-register all available plugins to global registry.

    Call this at module initialization to populate the registry.
    """
    registry = get_registry()

    # Only discover if registry is empty
    if not registry.list_plugins():
        discover_all(registry)
        logger.info(f"Auto-registered {len(registry.list_plugins())} plugins")

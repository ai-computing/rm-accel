"""
Auto-hook module for transparent backend switching.

Provides two usage patterns:

1. Simple auto mode (uses environment variables):
    import rm_abstract.auto
    from vllm import LLM  # Automatically routed

2. Explicit configuration:
    from rm_abstract.auto import use
    use(device="rbln", engine="vllm")
    from vllm import LLM  # Uses Rebellions NPU with vLLM

Configuration Options:
    device: cpu, nvidia, rbln, furiosa, auto
    engine: vllm, triton, torchserve
"""

import sys
import os
import logging
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

# Global configuration
_config = {
    "device": os.environ.get("RM_DEVICE", "auto"),
    "engine": os.environ.get("RM_ENGINE", "vllm"),
    "hook_vllm": os.environ.get("RM_HOOK_VLLM", "1") == "1",
    "hook_transformers": os.environ.get("RM_HOOK_TRANSFORMERS", "1") == "1",
}


def use(
    device: Optional[str] = None,
    engine: Optional[str] = None,
    hook_vllm: Optional[bool] = None,
    hook_transformers: Optional[bool] = None,
) -> None:
    """
    Configure RM Abstract before importing ML libraries.

    This function MUST be called before importing vllm or transformers.

    Args:
        device: Device selection
            - "auto": Auto-detect best available (NPU > GPU > CPU)
            - "cpu": CPU with PyTorch
            - "nvidia" or "gpu": NVIDIA GPU with vLLM
            - "rbln": Rebellions ATOM NPU
            - "furiosa": FuriosaAI RNGD NPU
        engine: Serving engine
            - "vllm": vLLM (default)
            - "triton": Triton Inference Server
            - "torchserve": TorchServe
        hook_vllm: Enable/disable vLLM import hooking
        hook_transformers: Enable/disable transformers import hooking

    Example:
        from rm_abstract.auto import use
        use(device="rbln", engine="vllm")

        from vllm import LLM, SamplingParams
        llm = LLM(model="gpt2")  # Runs on Rebellions NPU
    """
    global _config

    if device is not None:
        _config["device"] = device
        os.environ["RM_DEVICE"] = device

    if engine is not None:
        _config["engine"] = engine
        os.environ["RM_ENGINE"] = engine

    if hook_vllm is not None:
        _config["hook_vllm"] = hook_vllm
        os.environ["RM_HOOK_VLLM"] = "1" if hook_vllm else "0"

    if hook_transformers is not None:
        _config["hook_transformers"] = hook_transformers
        os.environ["RM_HOOK_TRANSFORMERS"] = "1" if hook_transformers else "0"

    # Re-enable hooks with new config
    disable_hooks()
    enable_hooks()

    logger.info(f"RM Abstract configured: device={_config['device']}, engine={_config['engine']}")


def get_config() -> dict:
    """Get current configuration."""
    return _config.copy()


class _VLLMCompatModule(ModuleType):
    """Proxy module for vllm that routes to rm_abstract.compat."""

    def __init__(self):
        super().__init__("vllm")
        self.__path__ = []
        self.__file__ = __file__

    def __getattr__(self, name):
        from rm_abstract.compat import vllm_compat
        return getattr(vllm_compat, name)


class _TransformersCompatModule(ModuleType):
    """Proxy module for transformers that routes to rm_abstract.compat."""

    def __init__(self):
        super().__init__("transformers")
        self.__path__ = []
        self.__file__ = __file__
        self._original_transformers = None

    def _get_original(self):
        """Lazily load original transformers module."""
        if self._original_transformers is None:
            for finder in sys.meta_path[:]:
                if isinstance(finder, RMAbstractFinder):
                    sys.meta_path.remove(finder)
                    try:
                        import importlib
                        self._original_transformers = importlib.import_module("transformers")
                    finally:
                        sys.meta_path.insert(0, finder)
                    break
        return self._original_transformers

    def __getattr__(self, name):
        if name in ("AutoModelForCausalLM", "AutoTokenizer"):
            from rm_abstract.compat import transformers_compat
            return getattr(transformers_compat, name)

        original = self._get_original()
        if original:
            return getattr(original, name)

        raise AttributeError(f"module 'transformers' has no attribute '{name}'")


class RMAbstractLoader(Loader):
    """Loader for hooked modules."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def create_module(self, spec):
        if self.module_name == "vllm":
            return _VLLMCompatModule()
        elif self.module_name == "transformers":
            return _TransformersCompatModule()
        return None

    def exec_module(self, module):
        pass


class RMAbstractFinder(MetaPathFinder):
    """Meta path finder that intercepts imports of vLLM and transformers."""

    def __init__(self):
        self.hooked_modules = set()
        if _config["hook_vllm"]:
            self.hooked_modules.add("vllm")
        if _config["hook_transformers"]:
            self.hooked_modules.add("transformers")

    def find_spec(self, fullname, path, target=None):
        root_module = fullname.split(".")[0]

        if root_module in self.hooked_modules:
            if fullname == root_module:
                logger.debug(f"Hooking import: {fullname}")
                return ModuleSpec(fullname, RMAbstractLoader(fullname))

        return None


def enable_hooks():
    """Enable import hooks for vLLM and transformers."""
    for finder in sys.meta_path:
        if isinstance(finder, RMAbstractFinder):
            return

    finder = RMAbstractFinder()
    sys.meta_path.insert(0, finder)
    logger.debug(
        f"RM Abstract hooks enabled: device={_config['device']}, "
        f"engine={_config['engine']}, modules={finder.hooked_modules}"
    )


def disable_hooks():
    """Disable import hooks."""
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RMAbstractFinder)]
    # Clear cached modules
    for mod in ["vllm", "transformers"]:
        if mod in sys.modules:
            del sys.modules[mod]


def is_hooked() -> bool:
    """Check if hooks are currently enabled."""
    return any(isinstance(f, RMAbstractFinder) for f in sys.meta_path)


# Auto-enable hooks when this module is imported
enable_hooks()

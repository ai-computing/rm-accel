"""
Auto-hook module for transparent backend switching.

Import this module to automatically intercept vLLM and transformers imports
and route them to the appropriate backend based on RM_DEVICE environment variable.

Usage:
    # Add this single line at the top of your script
    import rm_abstract.auto

    # Then use vLLM or transformers as normal
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM

    # The imports will be automatically routed to rm_abstract.compat

Environment Variables:
    RM_DEVICE: Device selection (auto, cpu, nvidia, rbln, furiosa)
    RM_ENGINE: Serving engine (vllm, triton, torchserve)
    RM_HOOK_VLLM: Enable/disable vLLM hooking (1/0, default: 1)
    RM_HOOK_TRANSFORMERS: Enable/disable transformers hooking (1/0, default: 1)
"""

import sys
import os
import logging
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from types import ModuleType

logger = logging.getLogger(__name__)

# Configuration from environment
HOOK_VLLM = os.environ.get("RM_HOOK_VLLM", "1") == "1"
HOOK_TRANSFORMERS = os.environ.get("RM_HOOK_TRANSFORMERS", "1") == "1"


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
            # Temporarily remove our hook to import real transformers
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
        # Route specific classes to our compat layer
        if name in ("AutoModelForCausalLM", "AutoTokenizer"):
            from rm_abstract.compat import transformers_compat
            return getattr(transformers_compat, name)

        # Everything else goes to original transformers
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
    """
    Meta path finder that intercepts imports of vLLM and transformers.

    When these modules are imported, this finder returns our compatibility
    layer instead of the original modules.
    """

    HOOKED_MODULES = set()

    def __init__(self):
        if HOOK_VLLM:
            self.HOOKED_MODULES.add("vllm")
        if HOOK_TRANSFORMERS:
            self.HOOKED_MODULES.add("transformers")

    def find_spec(self, fullname, path, target=None):
        # Only hook top-level module imports
        root_module = fullname.split(".")[0]

        if root_module in self.HOOKED_MODULES:
            # For submodule imports (e.g., vllm.sampling_params),
            # let the parent module handle it
            if fullname == root_module:
                logger.debug(f"Hooking import: {fullname}")
                return ModuleSpec(fullname, RMAbstractLoader(fullname))

        return None


def enable_hooks():
    """Enable import hooks for vLLM and transformers."""
    # Check if already installed
    for finder in sys.meta_path:
        if isinstance(finder, RMAbstractFinder):
            return

    finder = RMAbstractFinder()
    sys.meta_path.insert(0, finder)
    logger.info(
        f"RM Abstract hooks enabled. "
        f"Device: {os.environ.get('RM_DEVICE', 'auto')}, "
        f"Hooked modules: {finder.HOOKED_MODULES}"
    )


def disable_hooks():
    """Disable import hooks."""
    sys.meta_path = [f for f in sys.meta_path if not isinstance(f, RMAbstractFinder)]
    logger.info("RM Abstract hooks disabled")


def is_hooked() -> bool:
    """Check if hooks are currently enabled."""
    return any(isinstance(f, RMAbstractFinder) for f in sys.meta_path)


# Auto-enable hooks when this module is imported
enable_hooks()

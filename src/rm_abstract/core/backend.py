"""
Backend Abstract Base Class

Defines the interface that all device backends (GPU, NPU, CPU) must implement.

Backend vs ServingEngine:
------------------------
- Backend: Hardware abstraction for running inference on specific devices
  - Manages device initialization and cleanup
  - Prepares models (compilation, optimization)
  - Executes inference requests
  - Used by DeviceFlowController for transparent device switching

- ServingEngine: Serving infrastructure abstraction
  - Manages server lifecycle (start, stop, health check)
  - Provides HTTP/gRPC API
  - Handles batching, scheduling
  - See rm_abstract/serving/ for ServingEngine implementations

See rm_abstract/ARCHITECTURE.md for detailed explanation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from enum import Enum


class DeviceType(Enum):
    """Device type enumeration"""

    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"
    AUTO = "auto"


@dataclass
class DeviceInfo:
    """Device information"""

    device_type: DeviceType
    device_id: int
    name: str
    vendor: Optional[str] = None
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    extra: Optional[Dict[str, Any]] = None


class Backend(ABC):
    """
    Backend Abstract Base Class

    Common interface that all device backends must implement.
    """

    def __init__(self, device_id: int = 0, **kwargs):
        """
        Args:
            device_id: Device index (0, 1, 2, ...)
            **kwargs: Additional backend-specific options
        """
        self.device_id = device_id
        self.options = kwargs
        self._initialized = False

    @property
    @abstractmethod
    def device_type(self) -> DeviceType:
        """Return device type"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return backend name (e.g., 'CUDA', 'RBLN', 'Furiosa')"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend is available

        Returns:
            True if available
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the backend

        Performs device connection, runtime initialization, etc.
        """
        ...

    @abstractmethod
    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model for execution

        - GPU: Move model to GPU (passthrough)
        - NPU: ONNX conversion → NPU compilation → return compiled model
        - CPU: Return model as-is (passthrough)

        Args:
            model: PyTorch model or model path
            model_config: Model configuration (batch size, precision, etc.)

        Returns:
            Prepared model (ready for execution on the backend)
        """
        ...

    @abstractmethod
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference

        Args:
            model: Model prepared by prepare_model()
            inputs: Input data
            **kwargs: Additional execution options

        Returns:
            Inference result
        """
        ...

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """
        Return device information

        Returns:
            DeviceInfo object
        """
        ...

    def cleanup(self) -> None:
        """
        Cleanup backend

        Release resources, clear memory, etc.
        """
        self._initialized = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device_id={self.device_id})"


class NPUBackend(Backend, ABC):
    """
    NPU Backend Common Base Class

    Includes NPU-specific features (compilation, caching).
    """

    def __init__(
        self,
        device_id: int = 0,
        cache_dir: Optional[str] = None,
        compile_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(device_id, **kwargs)
        self.cache_dir = cache_dir
        self.compile_options = compile_options or {}
        self._compiled_models: Dict[str, Any] = {}

    @property
    def device_type(self) -> DeviceType:
        return DeviceType.NPU

    @abstractmethod
    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        Compile model for NPU

        Args:
            model: ONNX model or PyTorch model
            **kwargs: Compilation options

        Returns:
            Compiled model
        """
        ...

    @abstractmethod
    def load_compiled_model(self, path: str) -> Any:
        """
        Load compiled model

        Args:
            path: Compiled model file path

        Returns:
            Loaded compiled model
        """
        ...

    @abstractmethod
    def save_compiled_model(self, model: Any, path: str) -> None:
        """
        Save compiled model

        Args:
            model: Compiled model
            path: Save path
        """
        ...

    def get_cache_key(self, model: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate model cache key

        Args:
            model: Model object
            config: Model configuration

        Returns:
            Cache key string
        """
        import hashlib

        # Generate hash from model name/path and configuration
        model_name = getattr(model, "name_or_path", str(type(model).__name__))
        config_str = str(sorted(config.items())) if config else ""

        key_string = f"{model_name}_{self.name}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

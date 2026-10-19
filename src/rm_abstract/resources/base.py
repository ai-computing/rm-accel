"""
Base Resource Class

Defines the common interface for all hardware resources.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Resource type enumeration"""

    GPU = "gpu"
    NPU = "npu"
    CPU = "cpu"


@dataclass
class ResourceCapability:
    """Resource capability information"""

    # Supported features
    supports_fp16: bool = True
    supports_fp32: bool = True
    supports_int8: bool = False
    supports_int4: bool = False

    # Performance characteristics
    max_batch_size: Optional[int] = None
    max_sequence_length: Optional[int] = None

    # Special features
    supports_continuous_batching: bool = False
    supports_tensor_parallel: bool = False

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceInfo:
    """Resource information container"""

    resource_type: ResourceType
    device_id: int
    name: str
    vendor: str
    memory_total: Optional[int] = None  # bytes
    memory_available: Optional[int] = None  # bytes
    capability: Optional[ResourceCapability] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Resource(ABC):
    """
    Base Resource Class

    All hardware resources (GPU, NPU, CPU) implement this interface.
    This provides a unified way to interact with different hardware.

    Example:
        # Using a resource
        resource = NvidiaGPU(device_id=0)

        if resource.is_available():
            resource.initialize()
            prepared_model = resource.prepare_model(model)
            output = resource.execute(prepared_model, inputs)
            resource.cleanup()
    """

    # Resource metadata (override in subclasses)
    RESOURCE_TYPE: ResourceType = ResourceType.CPU
    VENDOR: str = "generic"
    NAME: str = "Generic Resource"

    def __init__(self, device_id: int = 0, **kwargs):
        """
        Args:
            device_id: Device index (0, 1, 2, ...)
            **kwargs: Additional resource-specific options
        """
        self.device_id = device_id
        self.options = kwargs
        self._initialized = False

    @property
    def resource_type(self) -> ResourceType:
        """Return resource type"""
        return self.RESOURCE_TYPE

    @property
    def vendor(self) -> str:
        """Return vendor name"""
        return self.VENDOR

    @property
    def name(self) -> str:
        """Return resource name"""
        return self.NAME

    @property
    def is_initialized(self) -> bool:
        """Check if resource is initialized"""
        return self._initialized

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this resource is available.

        Returns:
            True if the resource is available and can be used
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the resource.

        Performs device connection, runtime initialization, etc.
        """
        ...

    @abstractmethod
    def prepare_model(self, model: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model for execution on this resource.

        Args:
            model: Model to prepare (PyTorch model, path, etc.)
            config: Model configuration options

        Returns:
            Prepared model ready for execution
        """
        ...

    @abstractmethod
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference on this resource.

        Args:
            model: Prepared model
            inputs: Input data
            **kwargs: Additional execution options

        Returns:
            Inference result
        """
        ...

    @abstractmethod
    def get_info(self) -> ResourceInfo:
        """
        Get resource information.

        Returns:
            ResourceInfo with details about this resource
        """
        ...

    def cleanup(self) -> None:
        """
        Cleanup resource.

        Release resources, clear memory, etc.
        Override in subclasses for specific cleanup logic.
        """
        self._initialized = False
        logger.debug(f"Resource {self.name} cleaned up")

    def get_capability(self) -> ResourceCapability:
        """
        Get resource capabilities.

        Returns:
            ResourceCapability with supported features
        """
        return ResourceCapability()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device_id={self.device_id})"

    def __enter__(self):
        """Context manager entry"""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


class NPUResource(Resource):
    """
    Base class for NPU resources.

    Extends Resource with NPU-specific features like compilation and caching.
    """

    RESOURCE_TYPE = ResourceType.NPU

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
    @abstractmethod
    def compiled_model_extension(self) -> str:
        """Return compiled model file extension (e.g., 'rbln', 'enf')"""
        ...

    @abstractmethod
    def compile_model(self, model: Any, **kwargs) -> Any:
        """
        Compile model for this NPU.

        Args:
            model: Model to compile (typically ONNX)
            **kwargs: Compilation options

        Returns:
            Compiled model
        """
        ...

    @abstractmethod
    def load_compiled_model(self, path: str) -> Any:
        """
        Load a compiled model from disk.

        Args:
            path: Path to compiled model file

        Returns:
            Loaded compiled model
        """
        ...

    @abstractmethod
    def save_compiled_model(self, model: Any, path: str) -> None:
        """
        Save a compiled model to disk.

        Args:
            model: Compiled model
            path: Path to save to
        """
        ...

    def get_capability(self) -> ResourceCapability:
        """NPU-specific capabilities"""
        return ResourceCapability(
            supports_fp16=True,
            supports_int8=True,
        )

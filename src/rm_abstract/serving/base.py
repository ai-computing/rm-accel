"""
Serving Engine Base Classes

Abstract base class for all serving engines (vLLM, Triton, TorchServe, etc.)

Backend vs ServingEngine:
------------------------
- Backend (rm_abstract/backends/): Hardware abstraction
  - Runs inference on specific devices (GPU, CPU, NPU)
  - Used with rm_abstract.init() for transparent device switching
  - Best for: local development, Jupyter notebooks, device experiments

- ServingEngine (this module): Serving infrastructure abstraction
  - Manages server lifecycle (start, stop, health check)
  - Provides HTTP/gRPC API for production deployment
  - Supports: vLLM, Triton Inference Server, TorchServe
  - Best for: production deployment, REST API serving

See rm_abstract/ARCHITECTURE.md for detailed explanation.

Example:
    from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType
    
    config = ServingConfig(engine=ServingEngineType.VLLM, port=8000)
    with create_serving_engine(config) as engine:
        engine.load_model("gpt2")
        output = engine.infer("Hello, I am")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

# For Python 3.9 compatibility with dataclass Optional fields
from typing import get_type_hints

logger = logging.getLogger(__name__)


class ServingEngineType(str, Enum):
    """Supported serving engine types"""
    VLLM = "vllm"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    RAY_SERVE = "ray_serve"
    

class DeviceTarget(str, Enum):
    """Target device for serving"""
    GPU = "gpu"
    NPU_RBLN = "npu_rbln"  # Rebellions ATOM
    NPU_FURIOSA = "npu_furiosa"  # FuriosaAI
    CPU = "cpu"


@dataclass
class ServingConfig:
    """Configuration for serving engine"""
    
    # Engine selection
    engine: ServingEngineType = ServingEngineType.VLLM
    device: DeviceTarget = DeviceTarget.GPU
    device_id: int = 0
    
    # Model configuration
    model_name: str = ""
    model_path: Optional[str] = None
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Performance options
    tensor_parallel_size: int = 1
    max_batch_size: int = 32
    max_seq_len: Optional[int] = None  # None = use model default
    
    # Additional engine-specific options
    extra_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "engine": self.engine.value,
            "device": self.device.value,
            "device_id": self.device_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            **self.extra_options,
        }


class ServingEngine(ABC):
    """
    Abstract base class for model serving engines
    
    All serving engines (vLLM, Triton, TorchServe) implement this interface.
    """
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self._server = None
        self._is_running = False
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name"""
        ...
    
    @property
    @abstractmethod
    def engine_type(self) -> ServingEngineType:
        """Engine type"""
        ...
    
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this serving engine is available"""
        ...
    
    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """List of supported devices"""
        return [DeviceTarget.GPU, DeviceTarget.CPU]
    
    @abstractmethod
    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load model for serving
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional options
            
        Returns:
            Loaded model
        """
        ...
    
    @abstractmethod
    def start_server(self) -> None:
        """Start the serving server"""
        ...
    
    @abstractmethod
    def stop_server(self) -> None:
        """Stop the serving server"""
        ...
    
    @property
    def is_running(self) -> bool:
        """Check if server is running"""
        return self._is_running
    
    @abstractmethod
    def infer(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference
        
        Args:
            inputs: Input data (text, tokens, etc.)
            **kwargs: Additional options
            
        Returns:
            Inference result
        """
        ...
    
    def health_check(self) -> bool:
        """Check server health"""
        return self._is_running
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get serving metrics"""
        return {
            "engine": self.name,
            "is_running": self._is_running,
            "config": self.config.to_dict(),
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_server()


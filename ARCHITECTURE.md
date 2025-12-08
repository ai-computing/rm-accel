# System Architecture

RM Abstract Layer architecture and design principles.

---

## Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (User Code, Inference Scripts)                  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   RM Abstract Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Plugin     │  │   Device     │  │   Hardware       │   │
│  │   Registry   │  │  Controller  │  │   Detector       │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         └─────────────────┼────────────────────┘             │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Resource / Engine Layer                   │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   GPU Resource   │  │   CPU Resource   │  │   NPU Resource   │
│    (NvidiaGPU)   │  │  (CPUResource)   │  │ (RebellionsATOM) │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   NVIDIA GPU     │  │      CPU         │  │  Rebellions NPU  │
│     (vLLM)       │  │   (PyTorch)      │  │  (vLLM-RBLN)     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Core Components

### 1. Plugin Registry

Centralized registration and discovery system for resources and engines.

```python
# registry/plugin_registry.py

class PluginRegistry:
    """Singleton registry for all plugins"""

    def register(self, plugin_info: PluginInfo) -> None:
        """Register a plugin"""

    def get(self, name: str) -> Optional[Type]:
        """Get plugin class by name"""

    def list_plugins(self, plugin_type: PluginType) -> List[PluginInfo]:
        """List plugins by type"""

    def get_best_available(self, plugin_type: PluginType) -> Optional[PluginInfo]:
        """Get best available plugin by priority"""
```

### 2. DeviceFlowController

Device management and resource routing.

```python
# core/controller.py

class DeviceFlowController:
    """Device flow controller"""

    def __init__(self, config: Config):
        self._resource = None
        self._registry = get_registry()
        self._initialize_resource()

    def switch_device(self, device: str) -> None:
        """Switch device at runtime"""

    def prepare_model(self, model, config=None) -> Any:
        """Prepare model for the resource"""

    def execute(self, model, inputs, **kwargs) -> Any:
        """Execute inference"""
```

### 3. Resource Interface

Base class for all hardware resources.

```python
# resources/base.py

class Resource(ABC):
    """Abstract base class for resources"""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if resource is available"""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize resource"""

    @abstractmethod
    def prepare_model(self, model, config=None) -> Any:
        """Prepare model"""

    @abstractmethod
    def execute(self, model, inputs, **kwargs) -> Any:
        """Execute inference"""

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
```

---

## Module Structure

```text
src/rm_abstract/
├── __init__.py           # Main interface
├── exceptions.py         # Exception hierarchy
│
├── common/               # Shared utilities
│   ├── input_adapter.py  # Input format unification
│   ├── output_adapter.py # Output format conversion
│   ├── model_converter.py # ONNX conversion
│   └── hardware_detector.py # Hardware detection
│
├── resources/            # Hardware resources
│   ├── base.py           # Resource ABC
│   ├── gpu/
│   │   └── nvidia.py     # NVIDIA GPU (vLLM)
│   ├── npu/
│   │   ├── rebellions.py # Rebellions ATOM
│   │   └── furiosa.py    # FuriosaAI RNGD
│   └── cpu/
│       └── default.py    # CPU fallback
│
├── engines/              # Serving engines
│   ├── base.py           # ServingEngine base
│   ├── vllm.py           # vLLM engine
│   ├── triton.py         # Triton engine
│   └── torchserve.py     # TorchServe engine
│
├── registry/             # Plugin system
│   ├── plugin_registry.py # Registry singleton
│   └── discovery.py      # Auto-discovery
│
└── core/                 # Core modules
    ├── config.py         # Configuration
    └── controller.py     # DeviceFlowController
```

---

## Resource Priority

Auto-selection uses priority-based selection:

| Priority | Resource Type | Example |
|----------|---------------|---------|
| 100 | NPU | Rebellions ATOM, FuriosaAI RNGD |
| 50 | GPU | NVIDIA GPU (vLLM) |
| 0 | CPU | CPU fallback (PyTorch) |

---

## Data Flow

### Initialization Flow

```text
1. rm_abstract.init(device="auto")
         │
         ▼
2. DeviceFlowController.__init__()
         │
         ▼
3. discover_resources(registry)
         │
         ▼
4. registry.get_best_available()
         │
         ▼
5. resource.initialize()
```

### Device Switch Flow

```text
1. rm_abstract.switch_device("nvidia")
         │
         ▼
2. controller.switch_device("nvidia")
         │
         ▼
3. old_resource.cleanup()
         │
         ▼
4. new_resource = NvidiaGPU()
         │
         ▼
5. new_resource.initialize()
```

---

## Design Principles

1. **Modularity**: Resources and engines are separate, pluggable modules
2. **Transparency**: Device details abstracted from user code
3. **Extensibility**: New resources via plugin registration
4. **Flexibility**: Runtime device switching
5. **Priority-based**: Automatic best-device selection

---

## Adding New Resources

### 1. Create Resource Class

```python
# resources/new_vendor/new_resource.py

from ..base import Resource, ResourceType, ResourceInfo

class NewResource(Resource):
    RESOURCE_TYPE = ResourceType.NPU
    VENDOR = "new_vendor"
    NAME = "NewResource"

    @staticmethod
    def is_available() -> bool:
        try:
            import new_sdk
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        self._initialized = True

    def prepare_model(self, model, config=None):
        return model

    def execute(self, model, inputs, **kwargs):
        return outputs

    def cleanup(self) -> None:
        self._initialized = False
```

### 2. Register in Discovery

```python
# registry/discovery.py

def discover_resources(registry: PluginRegistry) -> None:
    # ... existing resources ...

    # New resource
    try:
        from ..resources.new_vendor.new_resource import NewResource
        registry.register(PluginInfo(
            name="new",
            plugin_type=PluginType.RESOURCE,
            cls=NewResource,
            vendor="new_vendor",
            priority=100,  # NPU priority
        ))
    except ImportError:
        pass
```

---

## Related Documents

- [README.md](README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

# System Architecture

RM Abstract Layer architecture and design principles.

---

## Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│              (User Code, Inference Scripts)                  │
│                                                              │
│    from rm_abstract import LLM, SamplingParams              │
│    llm = LLM(model="gpt2")                                  │
│    outputs = llm.generate(["Hello"], SamplingParams())      │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLM Proxy Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Device     │  │   Engine     │  │   Backend        │   │
│  │   Selection  │  │   Selection  │  │   Initialization │   │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │
│         │                 │                    │             │
│         └─────────────────┼────────────────────┘             │
│                           │                                  │
│                           ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Backend Router                            │  │
│  │   vLLM | Triton | TorchServe | Transformers           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   GPU Backend    │  │   CPU Backend    │  │   NPU Backend    │
│   (vLLM/Triton)  │  │  (transformers)  │  │ (vLLM-RBLN/etc)  │
└────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
         │                     │                      │
         ▼                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   NVIDIA GPU     │  │      CPU         │  │  Rebellions NPU  │
│   (CUDA)         │  │   (PyTorch)      │  │  (FuriosaAI)     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Core Components

### 1. LLM Proxy Class

The main entry point providing vLLM-compatible API across all devices.

```python
# llm.py

class LLM:
    """Unified LLM interface for heterogeneous AI accelerators."""

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,   # auto, cpu, nvidia, rbln, furiosa
        engine: Optional[str] = None,   # vllm, triton, torchserve
        **kwargs,
    ):
        self._initialize_backend()

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[RequestOutput]:
        """Generate completions - same API as vLLM."""
```

### 2. Serving Engines

Auto-managed serving engines with model conversion.

```python
# engines/triton.py

class TritonEngine(ServingEngine):
    """Triton with auto-start and ONNX conversion."""

    def load_model(self, model_name):
        # 1. Convert HuggingFace model to ONNX
        # 2. Create Triton model repository
        # 3. Start Docker container
        # 4. Wait for server ready

# engines/torchserve.py

class TorchServeEngine(ServingEngine):
    """TorchServe with auto-start and MAR packaging."""

    def load_model(self, model_name):
        # 1. Convert HuggingFace model to MAR
        # 2. Start TorchServe server
        # 3. Register model
        # 4. Wait for model ready
```

### 3. Model Converter

Automatic model format conversion.

```python
# engines/model_converter.py

class ModelConverter:
    """Convert HuggingFace models to serving formats."""

    def to_onnx(self, model_name) -> str:
        """Convert to ONNX for Triton."""

    def to_mar(self, model_name) -> str:
        """Convert to MAR for TorchServe."""

    def create_triton_config(self, model_name, onnx_path) -> str:
        """Create Triton model repository structure."""
```

### 4. Plugin Registry (Classic API)

Centralized registration for resources.

```python
# registry/plugin_registry.py

class PluginRegistry:
    """Singleton registry for all plugins."""

    def register(self, plugin_info: PluginInfo) -> None
    def get(self, name: str) -> Optional[Type]
    def list_plugins(self, plugin_type: PluginType) -> List[PluginInfo]
    def get_best_available(self, plugin_type: PluginType) -> Optional[PluginInfo]
```

---

## Module Structure

```text
src/rm_abstract/
├── __init__.py           # Main interface (LLM, SamplingParams, init, etc.)
├── llm.py                # LLM proxy class (vLLM-compatible API)
├── exceptions.py         # Exception hierarchy
│
├── engines/              # Serving engines
│   ├── base.py           # ServingEngine base class
│   ├── vllm.py           # vLLM engine
│   ├── triton.py         # Triton engine (auto-start, ONNX conversion)
│   ├── torchserve.py     # TorchServe engine (auto-start, MAR packaging)
│   └── model_converter.py # HuggingFace to ONNX/MAR conversion
│
├── common/               # Shared utilities
│   ├── input_adapter.py  # Input format unification
│   ├── output_adapter.py # Output format conversion
│   └── hardware_detector.py # Hardware detection
│
├── resources/            # Hardware resources (classic API)
│   ├── base.py           # Resource ABC
│   ├── gpu/
│   │   └── nvidia.py     # NVIDIA GPU (vLLM)
│   ├── npu/
│   │   ├── rebellions.py # Rebellions ATOM
│   │   └── furiosa.py    # FuriosaAI RNGD
│   └── cpu/
│       └── default.py    # CPU fallback
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

## Backend Selection Flow

```text
LLM(model="gpt2", device="auto", engine="vllm")
         │
         ▼
┌─────────────────────────────────────┐
│     _initialize_backend()           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   _resolve_device("auto")           │
│   → Check: NPU > GPU > CPU          │
│   → Returns: "cpu" (if no GPU/NPU)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Route by engine:                  │
│   - "triton" → _init_triton_backend │
│   - "torchserve" → _init_torchserve │
│   - "vllm" → route by device        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Initialize backend:               │
│   - Load model                      │
│   - Start server (if needed)        │
│   - Set self._backend               │
└─────────────────────────────────────┘
```

---

## Auto Server Management

### Triton Flow

```text
LLM(model="gpt2", engine="triton")
         │
         ▼
┌─────────────────────────────────────┐
│  1. Check Docker available          │
│     → If not: fallback to           │
│       transformers                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Convert to ONNX                 │
│     → Cache: ~/.rm_abstract/models  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Create Triton repository        │
│     → config.pbtxt                  │
│     → model.onnx                    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Start Docker container          │
│     → nvcr.io/nvidia/tritonserver   │
│     → Mount model repository        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Wait for ready                  │
│     → HTTP health check             │
│     → localhost:8000/v2/health      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Register cleanup on exit        │
│     → atexit: stop container        │
└─────────────────────────────────────┘
```

### TorchServe Flow

```text
LLM(model="gpt2", engine="torchserve")
         │
         ▼
┌─────────────────────────────────────┐
│  1. Check TorchServe installed      │
│     → If not: fallback to           │
│       transformers                  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Convert to MAR                  │
│     → torch-model-archiver          │
│     → Cache: ~/.rm_abstract/models  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Start TorchServe                │
│     → torchserve --start            │
│     → config.properties             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Register model                  │
│     → Management API                │
│     → localhost:8081/models         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Wait for model ready            │
│     → Check worker status           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Register cleanup on exit        │
│     → atexit: torchserve --stop     │
└─────────────────────────────────────┘
```

---

## Resource Priority

Auto-selection uses priority-based selection:

| Priority | Resource Type | Example |
|----------|---------------|---------|
| 100 | NPU | Rebellions ATOM, FuriosaAI RNGD |
| 50 | GPU | NVIDIA GPU (vLLM) |
| 0 | CPU | CPU fallback (transformers) |

---

## Design Principles

1. **Zero Code Change**: Drop-in replacement for vLLM import
2. **Automatic Fallback**: Graceful degradation when resources unavailable
3. **Auto Management**: Server lifecycle handled automatically
4. **Caching**: Model conversions cached for fast subsequent runs
5. **Cleanup**: Resources released automatically on exit
6. **Extensibility**: New backends via plugin registration

---

## Adding New Backends

### 1. Create Engine Class

```python
# engines/new_engine.py

from .base import ServingEngine, ServingConfig

class NewEngine(ServingEngine):
    ENGINE_TYPE = ServingEngineType.NEW_ENGINE
    ENGINE_NAME = "NewEngine"

    @classmethod
    def is_available(cls) -> bool:
        # Check if dependencies available
        return True

    def load_model(self, model_name):
        # Prepare and load model
        pass

    def start_server(self):
        # Start serving server
        pass

    def infer(self, inputs, **kwargs):
        # Run inference
        pass
```

### 2. Add to LLM Proxy

```python
# llm.py

def _init_new_engine_backend(self, device: str):
    from .engines.new_engine import NewEngine
    # Initialize and configure engine
```

---

## Related Documents

- [README.md](../README.md) - Project overview
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide
- [ADDING_ENGINES.md](ADDING_ENGINES.md) - Adding new serving engines

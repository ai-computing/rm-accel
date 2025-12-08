# RM Abstract Layer

> **GPU / NPU / CPU Heterogeneous Accelerator Unified LLM Inference Abstraction Layer**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

RM Abstract Layer is an abstraction layer that enables LLM inference on various hardware accelerators (GPU, NPU, CPU) **without code modification**.

### Key Features

- **Device Transparency**: Existing code works as-is
- **Runtime Switching**: Real-time switching between GPU, CPU, NPU
- **Plugin Architecture**: Easily add new resources and engines via plugin registry
- **Multiple Serving Engines**: vLLM, Triton, TorchServe support
- **Hardware Detection**: Automatic detection and priority-based selection (NPU > GPU > CPU)

---

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# GPU support
pip install -e ".[gpu]"

# Full installation
pip install -e ".[all]"
```

### Basic Usage

```python
import rm_abstract

# Initialize (auto-select optimal device: NPU > GPU > CPU)
rm_abstract.init(device="auto")

# Check available resources
resources = rm_abstract.get_available_resources()
print(resources)
# {'nvidia': False, 'rbln': False, 'furiosa': False, 'cpu': True}

# Get device info
info = rm_abstract.get_device_info()
print(f"Current device: {info['resource_type']}:{info['device_id']}")
```

### Device Switching

```python
import rm_abstract

# Start with CPU
rm_abstract.init(device="cpu")

# Switch to GPU
rm_abstract.switch_device("nvidia")

# Switch to NPU
rm_abstract.switch_device("rbln")

# Check current device
info = rm_abstract.get_device_info()
print(f"Current: {info['resource_type']}:{info['device_id']}")
```

---

## Supported Environment

### Hardware Resources

| Device | Status | Resource Class |
|--------|--------|----------------|
| NVIDIA GPU | Supported | `NvidiaGPU` (vLLM) |
| CPU | Supported | `CPUResource` (PyTorch) |
| Rebellions ATOM NPU | Supported | `RebellionsATOM` (vLLM-RBLN / Optimum-RBLN) |
| FuriosaAI RNGD NPU | Supported | `FuriosaRNGD` |

### Serving Engines

| Engine | Status | Description |
|--------|--------|-------------|
| vLLM | Supported | High-performance LLM serving |
| Triton | Supported | Multi-model serving (Docker) |
| TorchServe | Supported | PyTorch native serving |

---

## Architecture

```
src/rm_abstract/
├── __init__.py           # Main interface (init, switch_device, etc.)
├── exceptions.py         # Exception classes
├── common/               # Shared utilities
│   ├── input_adapter.py  # Unified input format handling
│   ├── output_adapter.py # Unified output format conversion
│   ├── model_converter.py # ONNX conversion utilities
│   └── hardware_detector.py # Hardware detection
├── resources/            # Hardware-specific implementations
│   ├── base.py           # Resource ABC
│   ├── gpu/nvidia.py     # NVIDIA GPU (vLLM)
│   ├── npu/rebellions.py # Rebellions ATOM NPU
│   ├── npu/furiosa.py    # FuriosaAI RNGD NPU
│   └── cpu/default.py    # CPU fallback
├── engines/              # Serving engines
│   ├── base.py           # ServingEngine base class
│   ├── vllm.py           # vLLM engine
│   ├── triton.py         # Triton engine
│   └── torchserve.py     # TorchServe engine
├── registry/             # Plugin system
│   ├── plugin_registry.py # Centralized registry
│   └── discovery.py      # Auto-discovery
└── core/                 # Core modules
    ├── config.py         # Configuration
    └── controller.py     # DeviceFlowController
```

---

## Plugin Registry System

```python
from rm_abstract import get_registry, PluginType
from rm_abstract.registry import discover_all

# Get global registry
registry = get_registry()

# Discover all plugins
discover_all(registry)

# List available resources
for plugin in registry.list_plugins(PluginType.RESOURCE):
    status = "available" if plugin.is_available() else "not available"
    print(f"{plugin.name}: {status} (priority: {plugin.priority})")

# Get best available resource (by priority)
best = registry.get_best_available(PluginType.RESOURCE)
print(f"Best resource: {best.name}")
```

---

## Hardware Detection

```python
from rm_abstract import HardwareDetector

detector = HardwareDetector()

# Get summary
print(detector.summary())

# Check specific vendor
print(f"NVIDIA: {detector.is_available('nvidia')}")
print(f"Rebellions: {detector.is_available('rebellions')}")
print(f"CPU: {detector.is_available('cpu')}")

# Get best available hardware
best = detector.get_best_available()
print(f"Best: {best.hardware_type.value} - {best.name}")
```

---

## Serving Engines

```python
from rm_abstract.engines import (
    VLLMEngine,
    TritonEngine,
    TorchServeEngine,
    ServingConfig,
    ServingEngineType,
)

# vLLM Engine
if VLLMEngine.is_available():
    config = ServingConfig(
        engine=ServingEngineType.VLLM,
        model_name="gpt2",
    )
    engine = VLLMEngine(config)
    print(f"Engine: {engine.name}")
    print(f"Supported devices: {engine.supported_devices()}")
```

---

## Examples

```
examples/
├── basic_usage.py        # Basic init and resource check
├── device_switching.py   # Runtime device switching
├── hardware_detection.py # HardwareDetector usage
├── plugin_registry.py    # Plugin registry demo
└── serving_engine.py     # Serving engine demo
```

```bash
# Run examples
python examples/basic_usage.py
python examples/device_switching.py
python examples/hardware_detection.py
python examples/plugin_registry.py
python examples/serving_engine.py
```

---

## Testing

```bash
# Run all tests (57 tests)
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v
pytest tests/unit/test_registry.py -v
pytest tests/unit/test_resource.py -v
```

---

## Error Handling

```python
from rm_abstract.exceptions import (
    RMAbstractError,           # Base exception
    NotInitializedError,       # Not initialized
    BackendNotAvailableError,  # Backend unavailable
    InvalidDeviceError,        # Invalid device spec
)

try:
    rm_abstract.init(device="gpu:0")
except BackendNotAvailableError as e:
    print(f"GPU not available: {e}")
    rm_abstract.init(device="cpu")  # Fallback to CPU
```

---

## Device Options

```python
import rm_abstract

# Auto-select (NPU > GPU > CPU priority)
rm_abstract.init(device="auto")

# Specific GPU
rm_abstract.init(device="nvidia:0")
rm_abstract.init(device="gpu:0")  # alias

# CPU
rm_abstract.init(device="cpu")

# Rebellions NPU
rm_abstract.init(device="rbln:0")

# FuriosaAI NPU
rm_abstract.init(device="furiosa:0")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Quick start guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development/contribution guide |

---

## License

MIT License

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

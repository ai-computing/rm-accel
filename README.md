# RM Abstract Layer

> **GPU / NPU / CPU Heterogeneous Accelerator Unified LLM Inference Abstraction Layer**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

RM Abstract Layer is an abstraction layer that enables LLM inference on various hardware accelerators (GPU, NPU, CPU) **without code modification**.

### Key Features

- **Zero Code Change**: Just change `from vllm import LLM` to `from rm_abstract import LLM`
- **Device Transparency**: Existing vLLM code works as-is on any device
- **Runtime Switching**: Real-time switching between GPU, CPU, NPU
- **Auto Server Management**: Triton/TorchServe auto-start with model conversion
- **Plugin Architecture**: Easily add new resources and engines via plugin registry
- **Multiple Serving Engines**: vLLM, Triton, TorchServe support
- **Hardware Detection**: Automatic detection and priority-based selection (NPU > GPU > CPU)

---

## Quick Start

### Installation

```bash
# Basic installation (CPU with transformers)
pip install -r requirements/requirements.txt
pip install -e .

# GPU support (vLLM)
pip install -r requirements/requirements-gpu.txt
pip install -e ".[gpu]"

# Triton engine support (ONNX conversion)
pip install -r requirements/requirements-triton.txt
pip install -e ".[triton]"

# TorchServe engine support
pip install -r requirements/requirements-torchserve.txt
pip install -e ".[torchserve]"

# All serving engines
pip install -r requirements/requirements-serving.txt
pip install -e ".[serving]"

# Full installation (everything)
pip install -r requirements/requirements-all.txt
pip install -e ".[all]"
```

### Zero Code Change Usage (Recommended)

```python
# Before (vLLM only - GPU required)
# from vllm import LLM, SamplingParams

# After (rm_abstract - works on GPU, NPU, CPU)
from rm_abstract import LLM, SamplingParams

# Rest of your code remains EXACTLY the same!
llm = LLM(model="gpt2")
outputs = llm.generate(["Hello, I am"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

### Device Selection

```python
from rm_abstract import LLM, SamplingParams

# Auto-detect best device (NPU > GPU > CPU)
llm = LLM(model="gpt2")

# Explicit device selection
llm = LLM(model="gpt2", device="nvidia")   # NVIDIA GPU
llm = LLM(model="gpt2", device="cpu")      # CPU
llm = LLM(model="gpt2", device="rbln")     # Rebellions NPU
llm = LLM(model="gpt2", device="furiosa")  # FuriosaAI NPU

# Environment variable control
# RM_DEVICE=nvidia python your_script.py
# RM_DEVICE=cpu python your_script.py
```

### Engine Selection

```python
from rm_abstract import LLM, SamplingParams

# vLLM engine (default)
llm = LLM(model="gpt2", engine="vllm")

# Triton Inference Server (auto-starts Docker, converts to ONNX)
llm = LLM(model="gpt2", engine="triton")

# TorchServe (auto-starts server, creates MAR package)
llm = LLM(model="gpt2", engine="torchserve")

# Environment variable control
# RM_ENGINE=triton python your_script.py
```

### Classic API Usage

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

# Runtime device switching
rm_abstract.switch_device("cpu")
rm_abstract.switch_device("nvidia")  # If available
```

---

## Supported Environment

### Hardware Resources

| Device | Status | Backend |
|--------|--------|---------|
| NVIDIA GPU | Supported | vLLM / Triton / TorchServe |
| CPU | Supported | transformers / TorchServe |
| Rebellions ATOM NPU | Supported | vLLM-RBLN / Optimum-RBLN |
| FuriosaAI RNGD NPU | Supported | FuriosaAI SDK |

### Serving Engines

| Engine | Status | Auto Management | Description |
|--------|--------|-----------------|-------------|
| vLLM | Supported | - | High-performance LLM serving |
| Triton | Supported | Docker auto-start, ONNX conversion | Multi-model serving |
| TorchServe | Supported | Server auto-start, MAR packaging | PyTorch native serving |

---

## Architecture

```text
src/rm_abstract/
├── __init__.py           # Main interface (init, switch_device, LLM, SamplingParams)
├── llm.py                # LLM proxy class (vLLM-compatible API)
├── exceptions.py         # Exception classes
├── common/               # Shared utilities
│   ├── input_adapter.py  # Unified input format handling
│   └── output_adapter.py # Unified output format conversion
├── resources/            # Hardware-specific implementations
│   ├── base.py           # Resource ABC
│   ├── gpu/nvidia.py     # NVIDIA GPU (vLLM)
│   ├── npu/rebellions.py # Rebellions ATOM NPU
│   ├── npu/furiosa.py    # FuriosaAI RNGD NPU
│   └── cpu/default.py    # CPU fallback
├── engines/              # Serving engines
│   ├── base.py           # ServingEngine base class
│   ├── vllm.py           # vLLM engine
│   ├── triton.py         # Triton engine (auto-start, ONNX conversion)
│   ├── torchserve.py     # TorchServe engine (auto-start, MAR packaging)
│   └── model_converter.py # HuggingFace to ONNX/MAR conversion
├── registry/             # Plugin system
│   ├── plugin_registry.py # Centralized registry
│   └── discovery.py      # Auto-discovery
└── core/                 # Core modules
    ├── config.py         # Configuration
    └── controller.py     # DeviceFlowController
```

---

## Auto Server Management

### Triton Inference Server

When using `engine="triton"`, rm_abstract will automatically:

1. Convert HuggingFace model to ONNX format
2. Create Triton model repository with proper config.pbtxt
3. Start Docker container with Triton server
4. Clean up on exit

```python
from rm_abstract import LLM, SamplingParams

# Triton with auto-management (requires Docker)
llm = LLM(model="gpt2", engine="triton", device="nvidia")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

### TorchServe

When using `engine="torchserve"`, rm_abstract will automatically:

1. Convert HuggingFace model to MAR (Model Archive) format
2. Create TorchServe configuration
3. Start TorchServe server
4. Register and load the model
5. Clean up on exit

```python
from rm_abstract import LLM, SamplingParams

# TorchServe with auto-management (requires torchserve installed)
llm = LLM(model="gpt2", engine="torchserve", device="cpu")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

---

## Examples

All 5 examples are tested and validated (✅ 100% pass rate):

```
examples/
├── 01_quick_start.py         # Simplest usage
├── 02_device_selection.py    # Device control (CPU/GPU/NPU)
├── 03_auto_import_hook.py    # Auto import hooking
├── 04_serving_engines.py     # Test all serving engines
└── 05_engine_comparison.py   # Performance comparison
```

### Run Examples

```bash
# Quick start
PYTHONPATH=./src python examples/01_quick_start.py

# Device selection
PYTHONPATH=./src python examples/02_device_selection.py

# Auto import hook
PYTHONPATH=./src python examples/03_auto_import_hook.py

# Test all serving engines
PYTHONPATH=./src python examples/04_serving_engines.py

# Compare engine performance
PYTHONPATH=./src python examples/05_engine_comparison.py

# Run all examples with automated test
bash test_all_examples.sh
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_core.py -v
pytest tests/unit/test_llm.py -v
pytest tests/unit/test_registry.py -v
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
    from rm_abstract import LLM
    llm = LLM(model="gpt2", device="nvidia")
except BackendNotAvailableError as e:
    print(f"GPU not available: {e}")
    llm = LLM(model="gpt2", device="cpu")  # Fallback to CPU
```

---

## Device Options

```python
from rm_abstract import LLM

# Auto-select (NPU > GPU > CPU priority)
llm = LLM(model="gpt2", device="auto")

# Specific GPU
llm = LLM(model="gpt2", device="nvidia")
llm = LLM(model="gpt2", device="gpu")  # alias
llm = LLM(model="gpt2", device="cuda")  # alias

# CPU
llm = LLM(model="gpt2", device="cpu")

# Rebellions NPU
llm = LLM(model="gpt2", device="rbln")

# FuriosaAI NPU
llm = LLM(model="gpt2", device="furiosa")
```

---

## Documentation

### Getting Started
| Document | Description |
|----------|-------------|
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5-minute quick start guide |
| [docs/SERVING_ENGINES.md](docs/SERVING_ENGINES.md) | Complete guide for all serving engines |
| [docs/README_REQUIREMENTS.md](docs/README_REQUIREMENTS.md) | Installation and dependencies guide |

### Technical Documentation
| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture and design |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Project layout and organization |
| [docs/FINAL_TEST_REPORT.md](docs/FINAL_TEST_REPORT.md) | Test results and validation |
| [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) | Project cleanup summary |

### Development
| Document | Description |
|----------|-------------|
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | Development and contribution guide |
| [docs/ADDING_RESOURCES.md](docs/ADDING_RESOURCES.md) | Guide for adding new hardware resources |
| [docs/ADDING_ENGINES.md](docs/ADDING_ENGINES.md) | Guide for adding new serving engines |

### Presentation
| Document | Description |
|----------|-------------|
| [docs/PRESENTATION.md](docs/PRESENTATION.md) | Project presentation slides (Korean) |

---

## License

MIT License

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

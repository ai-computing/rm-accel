# RM Abstract Layer

> **GPU / NPU / CPU Heterogeneous Accelerator Unified LLM Inference Abstraction Layer**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

---

## Overview

RM Abstract Layer is an abstraction layer that enables LLM inference on various hardware accelerators (GPU, CPU) with **minimal code modification**.

### Key Features

- **Minimal Code Change**: Just change `from vllm import LLM` to `from rm_abstract import LLM`
- **Device Transparency**: vLLM-compatible API works on GPU and CPU
- **Runtime Switching**: Real-time switching between GPU and CPU
- **Auto Server Management**: Triton/TorchServe auto-start with model conversion
- **Plugin Architecture**: Easily add new resources and engines via plugin registry
- **Multiple Serving Engines**: vLLM, Triton, TorchServe support
- **Hardware Detection**: Automatic detection and priority-based selection (GPU > CPU)

---

## ai-rm

[ai-rm](https://github.com/ai-computing/ai-rm)의 **L1 가속기 추상화** 구성요소입니다(`components/l1-accel/rm_abstract_layer`). [rm-core protocol v0.1](https://github.com/ai-computing/rm-core/blob/v0.1.0/PROTOCOL.md)을 구현합니다.

- **역할**: 장치 탐지와 장치 능력 모델(`resources/base.py`의 `ResourceInfo`, `ResourceCapability`), 벤더별 자원 플러그인(`resources/gpu`, `resources/npu/furiosa.py`, `resources/npu/rebellions.py`, `resources/cpu`), 서빙 엔진 플러그인(vLLM, Triton, TorchServe).
- **adapter** (`rm_abstract.airm`): `AcceleratorProvider` `rm-abstract` (entry point `airm.accelerator_providers`).
  - 장치별로 벤더 관리 도구(`nvidia-smi`, `furiosa-smi`, `rbln-stat`, 읽기 전용)를 먼저 쓰고, 도구가 없으면 기존 `HardwareDetector`로 탐지해 `rm_core.Accelerator`로 변환합니다. CPU는 보고하지 않으며 장치가 없으면 `[]`입니다.
  - `id`는 `<노드>/<vendor>:<index>`이고 index는 장치 자체 번호(`npu3` → `furiosa:3`)입니다. 노드 이름은 `AIRM_NODE_NAME`, 없으면 hostname입니다.
  - `peak_tflops`는 벤더 데이터시트 값이 있는 모델(RNGD, ATOM)만 채우고 `peak_source`에 출처를 적습니다. NVIDIA GPU는 비워 둡니다(미상).
  - 생성자 인자: `node` (선택, 기본 `AIRM_NODE_NAME`/hostname).
- **노드 데몬**: `python -m rm_abstract.airm.daemon` — 주기적으로 탐지해 장치마다 `AcceleratorProfile`을 apply하고(연속으로 사라진 장치의 프로파일은 삭제), 노드 레이블 `airm.ai-computing.io/<vendor>.<model>=<개수>`, `airm.ai-computing.io/managed=true`를 패치합니다. `/metrics`에 `airm_accelerators{vendor,model}`을 냅니다.
  ```bash
  python -m rm_abstract.airm.daemon --once --dry-run        # 만들 객체만 출력 (클러스터 접근 없음)
  python -m rm_abstract.airm.daemon --provider sim --once --dry-run   # 장치 없이 시연
  pip install -e ".[k8s]"                                   # 실제 apply에는 kubernetes 클라이언트 필요
  ```
  환경 변수: `AIRM_NODE_NAME`, `AIRM_PROVIDER`(기본 `rm-abstract`), `AIRM_INTERVAL_S`(기본 60), `AIRM_METRICS_PORT`(기본 9400, 0이면 끔), `AIRM_REMOVE_AFTER`(기본 3: 연속 3회 탐지되지 않은 장치만 삭제 — 도구 일시 실패로 프로파일이 깜박이지 않도록).
- **k8s + llm-d**: Helm chart `deploy/helm/rm-abstract-node/` (DaemonSet + ServiceAccount/ClusterRole: `acceleratorprofiles` CRUD, `nodes` get/patch). AcceleratorProfile CRD는 rm-core(`k8s/acceleratorprofile.crd.yaml`)에 있고 ai-rm umbrella chart/llm-d-setup이 설치합니다. 이미지(`image.repository`)는 사이트 override로 지정합니다. 배치 정책(EPP)과 lmd-top은 이 프로파일·레이블을 읽습니다.
- **설치·테스트**: `ai-rm.yaml` 참고 (`pip install -e .`, `pytest -m "not gpu and not slow" tests/unit`, `python -m rm_core.conformance airm.accelerator_providers rm-abstract`).
- **지표**: `accelerators` (rm-core `METRICS`).
- **관련 성과지표**: 이종 자원 수(장치 등록·탐지).
- **`cluster.local.env` 변수**: 없음.

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

### Basic Usage

```python
# Before (vLLM only - GPU required)
# from vllm import LLM, SamplingParams

# After (rm_abstract - works on GPU and CPU)
from rm_abstract import LLM, SamplingParams

# Rest of your code remains the same!
llm = LLM(model="gpt2")
outputs = llm.generate(["Hello, I am"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

### Device Selection

```python
from rm_abstract import LLM, SamplingParams

# Auto-detect best device (GPU > CPU)
llm = LLM(model="gpt2")

# Explicit device selection
llm = LLM(model="gpt2", device="nvidia")   # NVIDIA GPU
llm = LLM(model="gpt2", device="cpu")      # CPU

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

# Initialize (auto-select optimal device: GPU > CPU)
rm_abstract.init(device="auto")

# Check available resources
resources = rm_abstract.get_available_resources()
print(resources)
# {'nvidia': True, 'cpu': True}

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
| NVIDIA GPU | ✅ Supported | vLLM / Triton / TorchServe |
| CPU | ✅ Supported | transformers / TorchServe |
| Rebellions ATOM NPU | 🔧 Planned | vLLM-RBLN / Optimum-RBLN |
| FuriosaAI RNGD NPU | 🔧 Planned | FuriosaAI SDK |

### Serving Engines

| Engine | Status | Auto Management | Description |
|--------|--------|-----------------|-------------|
| vLLM | ✅ Supported | - | High-performance LLM serving |
| Triton | ✅ Supported | Docker auto-start, ONNX conversion | Multi-model serving |
| TorchServe | ✅ Supported | Server auto-start, MAR packaging | PyTorch native serving |

---

## Architecture

```text
src/rm_abstract/
├── __init__.py           # Main interface (init, switch_device, LLM, SamplingParams)
├── llm.py                # LLM proxy class (vLLM-compatible API)
├── exceptions.py         # Exception classes
├── auto.py               # Auto import hooking (use() function)
├── run.py                # CLI runner
├── __main__.py           # Module entry point
├── airm/                 # ai-rm protocol v0.1 adapter (AcceleratorProvider, node daemon)
├── common/               # Shared utilities
│   ├── input_adapter.py  # Unified input format handling
│   ├── output_adapter.py # Unified output format conversion
│   ├── model_converter.py # Model format conversion
│   └── hardware_detector.py # Hardware detection
├── compat/               # Compatibility modules
│   ├── vllm_compat.py    # vLLM compatibility layer
│   └── transformers_compat.py # Transformers compatibility layer
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

```text
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

# Auto-select (GPU > CPU priority)
llm = LLM(model="gpt2", device="auto")

# Specific GPU
llm = LLM(model="gpt2", device="nvidia")
llm = LLM(model="gpt2", device="gpu")  # alias
llm = LLM(model="gpt2", device="cuda")  # alias

# CPU
llm = LLM(model="gpt2", device="cpu")

# NPU (Planned - not yet supported)
# llm = LLM(model="gpt2", device="rbln")     # Rebellions NPU
# llm = LLM(model="gpt2", device="furiosa")  # FuriosaAI NPU
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

BSD-3-Clause License

### Third-Party Licenses

This project integrates with various third-party libraries and SDKs:

#### Open Source Dependencies (Permissive Licenses)

| Package | License | Usage |
|---------|---------|-------|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | Deep learning framework |
| [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | High-performance LLM serving |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | BSD-3-Clause | Multi-model serving |
| [TorchServe](https://github.com/pytorch/serve) | Apache-2.0 | PyTorch model serving |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Model hub and inference |
| [vllm-rbln](https://github.com/rebellions-sw/vllm-rbln) | Apache-2.0 | vLLM plugin for RBLN NPU |
| [optimum-rbln](https://github.com/rebellions-sw/optimum-rbln) | Apache-2.0 | HuggingFace integration for RBLN |
| [FuriosaAI SDK](https://github.com/furiosa-ai/furiosa-sdk) | Apache-2.0 | FuriosaAI Python SDK |

#### Proprietary NPU SDKs

The following NPU backends require proprietary SDKs with separate license agreements:

| NPU | SDK | License | Access |
|-----|-----|---------|--------|
| **Rebellions ATOM** | RBLN SDK (rebel-compiler, drivers, runtime) | Proprietary | [Request Access](https://rebellions.ai/request-form-rbln-sdk/) |
| **FuriosaAI RNGD** | FuriosaAI SDK (kernel driver, firmware, runtime) | Proprietary (EULA) | [Early Access Program](https://furiosa.ai/) |

> **Note**: Users are responsible for complying with the respective SDK license terms when using NPU backends. This project only provides abstraction interfaces and does not redistribute any proprietary SDK components.

---

## Contributing

Contributions are welcome! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for details.

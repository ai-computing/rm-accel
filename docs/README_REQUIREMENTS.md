# Requirements Installation Guide

## Overview

RM Abstract Layer uses modular requirements files for flexible installation.

## Quick Reference

| File | Purpose | Includes |
|------|---------|----------|
| `requirements.txt` | Base (CPU) | torch, transformers, accelerate |
| `requirements-gpu.txt` | GPU support | + vLLM |
| `requirements-triton.txt` | Triton engine | + ONNX, tritonclient |
| `requirements-torchserve.txt` | TorchServe | + torchserve, model archiver |
| `requirements-serving.txt` | All engines | GPU + Triton + TorchServe |
| `requirements-dev.txt` | Development | + pytest, black, ruff, mypy |
| `requirements-all.txt` | Everything | Serving + Dev |

## Installation Examples

### Minimum (CPU only)

```bash
pip install -r requirements.txt
pip install -e .
```

**Includes**: torch, transformers, accelerate
**Use case**: CPU inference, testing, development

### GPU with vLLM

```bash
pip install -r requirements-gpu.txt
pip install -e ".[gpu]"
```

**Includes**: Base + vLLM
**Use case**: High-performance GPU inference

### Production (all engines)

```bash
pip install -r requirements-serving.txt
pip install -e ".[serving]"
```

**Includes**: vLLM + Triton + TorchServe
**Use case**: Production deployment with multiple backends

### Development

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

**Includes**: Base + testing tools + linters
**Use case**: Contributing, running tests

### Everything

```bash
pip install -r requirements-all.txt
pip install -e ".[all]"
```

**Includes**: All optional dependencies
**Use case**: Full-featured installation

## Dependency Tree

```
requirements.txt (BASE)
├── torch>=2.0.0
├── numpy>=1.20.0,<3.0.0
├── transformers>=4.30.0
├── accelerate>=0.20.0
├── sentencepiece>=0.1.99
└── protobuf>=3.20.0

requirements-gpu.txt
├── -r requirements.txt
└── vllm>=0.2.0

requirements-triton.txt
├── -r requirements.txt
├── onnx>=1.14.0
├── onnxscript>=0.1.0
├── tritonclient[http]>=2.34.0
└── requests>=2.28.0

requirements-torchserve.txt
├── -r requirements.txt
├── torchserve>=0.8.0
├── torch-model-archiver>=0.8.0
└── requests>=2.28.0

requirements-serving.txt
├── -r requirements-gpu.txt
├── -r requirements-triton.txt
└── -r requirements-torchserve.txt

requirements-dev.txt
├── -r requirements.txt
├── pytest>=7.0.0
├── pytest-cov>=4.0.0
├── pytest-asyncio>=0.21.0
├── black>=23.0.0
├── ruff>=0.1.0
├── mypy>=1.0.0
└── types-requests>=2.28.0

requirements-all.txt
├── -r requirements-serving.txt
└── -r requirements-dev.txt
```

## Special Cases

### Triton Only (no GPU)

```bash
# Base + Triton (ONNX support)
pip install -r requirements.txt
pip install -r requirements-triton.txt
pip install -e ".[triton]"
```

### TorchServe Only

```bash
# Base + TorchServe
pip install -r requirements.txt
pip install -r requirements-torchserve.txt
pip install -e ".[torchserve]"
```

### NPU Backends

NPU SDKs require separate installation from vendors:

**Rebellions ATOM**:
```bash
# Install Rebellions SDK first (from vendor)
# Then install rm_abstract
pip install -r requirements.txt
pip install -e ".[npu-rbln]"
```

**FuriosaAI RNGD**:
```bash
# Install FuriosaAI SDK first (from vendor)
# Then install rm_abstract
pip install -r requirements.txt
pip install -e ".[npu-furiosa]"
```

## Using UV (Recommended)

UV is a fast Python package installer:

```bash
# Install UV
pip install uv

# Install requirements with UV (much faster)
uv pip install -r requirements-serving.txt
uv pip install -e ".[serving]"
```

## Verification

After installation, verify:

```python
# Test basic import
python -c "from rm_abstract import LLM; print('✓ Base working')"

# Test GPU (if installed)
python -c "import vllm; print('✓ vLLM available')"

# Test Triton (if installed)
python -c "import tritonclient; print('✓ Triton client available')"

# Test TorchServe (if installed)
python -c "import torchserve; print('✓ TorchServe available')"
```

## Troubleshooting

### Torch CUDA Mismatch

If you get CUDA version errors:

```bash
# Install specific CUDA version
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### vLLM Installation Issues

```bash
# vLLM requires specific environment
pip install vllm --no-build-isolation
```

### TorchServe Not Found

```bash
# TorchServe is a separate binary
# Install via conda or from source
conda install -c pytorch torchserve
```

## Minimal Working Examples

### CPU Only

```bash
pip install torch transformers accelerate
pip install -e .
```

```python
from rm_abstract import LLM
llm = LLM(model="gpt2", device="cpu")
```

### GPU with vLLM

```bash
pip install torch transformers accelerate vllm
pip install -e .
```

```python
from rm_abstract import LLM
llm = LLM(model="gpt2", device="nvidia", enforce_eager=True)
```

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Installation](README.md#installation)
- [Examples](examples/)

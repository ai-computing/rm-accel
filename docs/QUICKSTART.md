# RM Abstract Layer - Quick Start Guide

Get started with RM Abstract Layer in 5 minutes!

## Installation

### Basic (CPU)

```bash
pip install -r requirements.txt
pip install -e .
```

### With GPU Support

```bash
pip install -r requirements-gpu.txt
pip install -e ".[gpu]"
```

### With All Serving Engines

```bash
pip install -r requirements-serving.txt
pip install -e ".[serving]"
```

## Hello World

```python
from rm_abstract import LLM, SamplingParams

# Auto-detect best device (NPU > GPU > CPU)
llm = LLM(model="gpt2")

# Generate text
outputs = llm.generate(
    ["Hello, I am"],
    SamplingParams(max_tokens=50, temperature=0.8)
)

print(outputs[0].outputs[0].text)
```

## Zero Code Change Migration

### Before (vLLM only - GPU required)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

### After (works on GPU, NPU, CPU)

```python
from rm_abstract import LLM, SamplingParams  # <- Only this line changed!

llm = LLM(model="gpt2")
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

## Device Selection

### Auto (Recommended)

```python
# Automatically selects: NPU > GPU > CPU
llm = LLM(model="gpt2", device="auto")
```

### Explicit Device

```python
# CPU
llm = LLM(model="gpt2", device="cpu")

# NVIDIA GPU
llm = LLM(model="gpt2", device="nvidia")

# Rebellions NPU
llm = LLM(model="gpt2", device="rbln")

# FuriosaAI NPU
llm = LLM(model="gpt2", device="furiosa")
```

## Serving Engines

### vLLM (Default)

```python
# High-performance GPU inference
llm = LLM(model="gpt2", engine="vllm", device="nvidia")
```

### TorchServe

```python
# Production deployment with auto-start
llm = LLM(model="gpt2", engine="torchserve")
```

### Triton Inference Server

```python
# ONNX optimization with Docker auto-start
llm = LLM(model="gpt2", engine="triton")
```

## Auto Import Hook

```python
# Configure BEFORE importing vllm
from rm_abstract.auto import use
use(device="cpu", hook_vllm=True)

# Now import vllm - automatically hooked!
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")  # Uses rm_abstract under the hood
```

## Examples

```bash
# Run examples
cd examples

# Quick start
python 01_quick_start.py

# Test all engines
PYTHONPATH=../src python 04_serving_engines.py

# Compare performance
PYTHONPATH=../src python 05_engine_comparison.py
```

## Common Patterns

### GPU with Memory Limit

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select GPU

llm = LLM(
    model="gpt2",
    device="nvidia",
    enforce_eager=True,  # Faster init
    gpu_memory_utilization=0.15,  # Low memory for shared GPU
)
```

### CPU Fallback

```python
from rm_abstract import LLM
from rm_abstract.exceptions import BackendNotAvailableError

try:
    llm = LLM(model="gpt2", device="nvidia")
except BackendNotAvailableError:
    print("GPU not available, using CPU")
    llm = LLM(model="gpt2", device="cpu")
```

### Batch Inference

```python
prompts = [
    "Hello, I am",
    "The future of AI",
    "Python is",
]

outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=50, temperature=0.8)
)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("-" * 80)
```

### Streaming (vLLM only)

```python
llm = LLM(model="gpt2", device="nvidia", engine="vllm")

for output in llm.generate(
    ["Hello, I am"],
    SamplingParams(max_tokens=100),
    use_tqdm=False,
):
    print(output.outputs[0].text, end="", flush=True)
```

## Troubleshooting

### GPU Hang

**Problem**: vLLM hangs on GPU initialization

**Solution**: Use eager mode
```python
llm = LLM(model="gpt2", enforce_eager=True)
```

### GPU Memory Error

**Problem**: CUDA out of memory

**Solution**: Reduce memory utilization
```python
llm = LLM(model="gpt2", gpu_memory_utilization=0.15)
```

### Import Hook Not Working

**Problem**: `from vllm import LLM` still uses original vLLM

**Solution**: Call `use()` BEFORE importing
```python
from rm_abstract.auto import use
use(device="cpu", hook_vllm=True)

# NOW import vllm
from vllm import LLM
```

## What's Next?

- [Serving Engines Guide](docs/SERVING_ENGINES.md) - Detailed engine comparison
- [Architecture](ARCHITECTURE.md) - System design
- [API Reference](docs/API.md) - Complete API documentation
- [Contributing](CONTRIBUTING.md) - Development guide

## Get Help

- **Issues**: https://github.com/gspark-etri/rm_abstract_layer/issues
- **Discussions**: https://github.com/gspark-etri/rm_abstract_layer/discussions

## Verified Environments

✅ **CPU**: All platforms with Python 3.9+
✅ **GPU**: NVIDIA GPUs with CUDA 11.8+
✅ **NPU**: Rebellions ATOM, FuriosaAI RNGD
✅ **Engines**: vLLM, TorchServe, Triton (all tested)

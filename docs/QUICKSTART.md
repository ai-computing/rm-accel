# Quick Start Guide

Get started with RM Abstract Layer in 5 minutes.

---

## Installation

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install
pip install -e ".[all]"
```

---

## Example 1: Zero Code Change (Recommended)

The simplest way to use RM Abstract Layer - just change one import line:

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

---

## Example 2: Device Selection

```python
from rm_abstract import LLM, SamplingParams

# Auto-detect best device (NPU > GPU > CPU)
llm = LLM(model="gpt2")
print(f"Backend: {llm.backend}")  # e.g., "transformers-cpu"

# Explicit device selection
llm = LLM(model="gpt2", device="nvidia")   # NVIDIA GPU
llm = LLM(model="gpt2", device="cpu")      # CPU
llm = LLM(model="gpt2", device="rbln")     # Rebellions NPU
llm = LLM(model="gpt2", device="furiosa")  # FuriosaAI NPU

# Generate
outputs = llm.generate(
    ["The future of AI is"],
    SamplingParams(temperature=0.7, max_tokens=50)
)
print(outputs[0].outputs[0].text)
```

---

## Example 3: Engine Selection

```python
from rm_abstract import LLM, SamplingParams

# vLLM engine (default)
llm = LLM(model="gpt2", engine="vllm")

# Triton Inference Server
# - Auto-converts model to ONNX
# - Auto-starts Docker container
# - Auto-cleans up on exit
llm = LLM(model="gpt2", engine="triton", device="nvidia")

# TorchServe
# - Auto-packages model to MAR format
# - Auto-starts TorchServe server
# - Auto-cleans up on exit
llm = LLM(model="gpt2", engine="torchserve", device="cpu")

# Generate
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

---

## Example 4: Environment Variables

Control device and engine via environment variables:

```bash
# Select device
RM_DEVICE=nvidia python your_script.py
RM_DEVICE=cpu python your_script.py
RM_DEVICE=rbln python your_script.py

# Select engine
RM_ENGINE=vllm python your_script.py
RM_ENGINE=triton python your_script.py
RM_ENGINE=torchserve python your_script.py
```

```python
# In your script - no changes needed
from rm_abstract import LLM, SamplingParams

llm = LLM(model="gpt2")  # Uses RM_DEVICE and RM_ENGINE from environment
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

---

## Example 5: Classic API (Device Switching)

```python
import rm_abstract

# Initialize (auto-select: NPU > GPU > CPU)
rm_abstract.init(device="auto")

# Check available resources
for name, available in rm_abstract.get_available_resources().items():
    status = "OK" if available else "Not available"
    print(f"  {name}: {status}")

# Get device info
info = rm_abstract.get_device_info()
print(f"Current device: {info['resource_type']}:{info['device_id']}")

# Switch device at runtime
rm_abstract.switch_device("cpu")
print(f"Switched to: {rm_abstract.get_device_info()}")
```

---

## Example 6: Plugin Registry

```python
from rm_abstract import get_registry, PluginType
from rm_abstract.registry import discover_all

# Get registry and discover plugins
registry = get_registry()
discover_all(registry)

# List resources with priority
print("Available Resources:")
for plugin in registry.list_plugins(PluginType.RESOURCE):
    status = "available" if plugin.is_available() else "not available"
    print(f"  {plugin.name} (priority: {plugin.priority}) [{status}]")

# Get best available
best = registry.get_best_available(PluginType.RESOURCE)
if best:
    print(f"Best resource: {best.name}")
```

---

## Device Options

```python
from rm_abstract import LLM

# Auto-select (NPU > GPU > CPU)
llm = LLM(model="gpt2", device="auto")

# Specific devices
llm = LLM(model="gpt2", device="nvidia")   # NVIDIA GPU
llm = LLM(model="gpt2", device="gpu")      # Alias for nvidia
llm = LLM(model="gpt2", device="cuda")     # Alias for nvidia
llm = LLM(model="gpt2", device="cpu")      # CPU
llm = LLM(model="gpt2", device="rbln")     # Rebellions NPU
llm = LLM(model="gpt2", device="furiosa")  # FuriosaAI NPU
```

---

## Error Handling

```python
from rm_abstract import LLM, SamplingParams
from rm_abstract.exceptions import (
    RMAbstractError,
    BackendNotAvailableError,
    NotInitializedError,
)

try:
    llm = LLM(model="gpt2", device="nvidia")
except BackendNotAvailableError:
    print("GPU not available, falling back to CPU")
    llm = LLM(model="gpt2", device="cpu")

outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
```

---

## FAQ

### Q: Model loads slowly on first run?

For Triton/TorchServe engines, the first run includes:

- Downloading model from HuggingFace (cached)
- Converting to ONNX/MAR format (cached at `~/.rm_abstract/models`)
- Starting server container

Subsequent runs use cached conversions and are much faster.

### Q: How to force CPU even with GPU available?

```python
llm = LLM(model="gpt2", device="cpu")
# or
# RM_DEVICE=cpu python your_script.py
```

### Q: How to check which backend is being used?

```python
llm = LLM(model="gpt2")
print(f"Backend: {llm.backend}")
print(f"Device: {llm.device}")
print(f"Engine: {llm.engine_type}")
```

### Q: Triton requires Docker but I don't have it?

rm_abstract will automatically fall back to transformers if Docker is not available:

```python
llm = LLM(model="gpt2", engine="triton")
# If Docker not available, falls back to transformers backend
print(llm.backend)  # "transformers-cpu" or "transformers-gpu"
```

---

## Running Examples

```bash
# Zero code change demo
python examples/zero_code_change.py

# Basic usage
python examples/basic_usage.py

# Test inference with CLI args
python examples/test_inference.py --device cpu --model gpt2

# Compare performance across devices
python examples/compare_devices.py
```

---

## Next Steps

- [README.md](../README.md) - Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

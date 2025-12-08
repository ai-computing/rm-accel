# Serving Engines Guide

RM Abstract Layer supports multiple serving engines for LLM inference. Each engine has different characteristics and use cases.

## Supported Engines

| Engine | Status | Auto-start | Best For |
|--------|--------|------------|----------|
| vLLM | ✅ Stable | - | GPU inference, high throughput |
| TorchServe | ✅ Stable | Yes | Production deployment, model management |
| Triton | ✅ Stable | Yes (Docker) | Multi-framework serving, ONNX optimization |

## Engine Selection

### Quick Start

```python
from rm_abstract import LLM, SamplingParams

# Default: vLLM (or transformers fallback on CPU)
llm = LLM(model="gpt2")

# Explicit engine selection
llm = LLM(model="gpt2", engine="vllm")      # vLLM
llm = LLM(model="gpt2", engine="torchserve") # TorchServe
llm = LLM(model="gpt2", engine="triton")     # Triton
```

### Environment Variable

```bash
export RM_ENGINE=torchserve
python your_script.py
```

## vLLM Engine

### Overview
- **Best for**: GPU inference with high throughput
- **Auto-start**: No (library-based)
- **Requirements**: CUDA-capable GPU

### Installation

```bash
pip install vllm>=0.2.0
```

### Usage

```python
from rm_abstract import LLM, SamplingParams

# GPU with vLLM
llm = LLM(
    model="gpt2",
    device="nvidia",
    engine="vllm",
    enforce_eager=True,  # Faster initialization
    gpu_memory_utilization=0.15,
)

outputs = llm.generate(
    ["Hello, I am"],
    SamplingParams(max_tokens=50, temperature=0.8)
)
```

### CPU Fallback

On CPU, vLLM automatically falls back to transformers:

```python
# CPU - automatically uses transformers
llm = LLM(model="gpt2", device="cpu", engine="vllm")
```

### Performance Tips

1. **Use eager mode for faster init**:
   ```python
   llm = LLM(model="gpt2", enforce_eager=True)
   ```

2. **Set GPU memory utilization**:
   ```python
   llm = LLM(model="gpt2", gpu_memory_utilization=0.5)
   ```

3. **Select specific GPU**:
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"
   ```

## TorchServe Engine

### Overview
- **Best for**: Production deployment, model versioning
- **Auto-start**: Yes (daemon mode)
- **Requirements**: `torchserve`, `torch-model-archiver`

### Installation

```bash
pip install torchserve torch-model-archiver
```

### Usage

```python
from rm_abstract import LLM, SamplingParams

# TorchServe with auto-start
llm = LLM(model="gpt2", engine="torchserve")

outputs = llm.generate(
    ["Hello, I am"],
    SamplingParams(max_tokens=50)
)
```

### Auto-Management

RM Abstract automatically:
1. Creates MAR (Model Archive) from HuggingFace model
2. Starts TorchServe server in daemon mode
3. Registers and loads the model
4. Handles cleanup on exit

### Configuration

MAR files and config are stored in `~/.rm_abstract/`:

```
~/.rm_abstract/
├── models/mar/          # Model archives
│   └── gpt2.mar
├── torchserve_models/   # Model store
└── torchserve_config.properties
```

### Manual Control

```python
from rm_abstract.engines.torchserve import TorchServeEngine

engine = TorchServeEngine(model_name="gpt2")
engine.start_server()  # Start manually
engine.stop_server()   # Stop manually
```

## Triton Inference Server

### Overview
- **Best for**: Multi-model serving, ONNX optimization
- **Auto-start**: Yes (Docker container)
- **Requirements**: Docker, `tritonclient`, ONNX tools

### Installation

```bash
# Install Python client
pip install tritonclient[http] onnx onnxscript

# Docker must be installed and running
docker --version
```

### Usage

```python
from rm_abstract import LLM, SamplingParams

# Triton with auto-start (requires Docker)
llm = LLM(model="gpt2", engine="triton")

outputs = llm.generate(
    ["Hello, I am"],
    SamplingParams(max_tokens=50)
)
```

### Auto-Management

RM Abstract automatically:
1. Converts HuggingFace model to ONNX format
2. Creates Triton model repository with `config.pbtxt`
3. Starts Docker container with Triton server
4. Waits for server readiness
5. Cleans up on exit

### Model Repository

ONNX models are stored in `~/.rm_abstract/`:

```
~/.rm_abstract/
├── models/onnx/
│   └── gpt2/
│       ├── model.onnx
│       └── config.pbtxt
└── triton_models/  # Triton model repository
```

### Manual Control

```python
from rm_abstract.engines.triton import TritonEngine

engine = TritonEngine(model_name="gpt2")
engine.start_docker()   # Start Triton server
engine.stop_docker()    # Stop server
```

## Engine Comparison

### Performance

Run the benchmark:

```bash
python examples/05_engine_comparison.py
```

Example results:

| Engine      | Init (s) | Inference (s) | Per prompt (s) |
|-------------|----------|---------------|----------------|
| vLLM        | 2.5      | 1.2           | 0.24           |
| TorchServe  | 5.3      | 1.8           | 0.36           |
| Triton      | 8.1      | 1.5           | 0.30           |

*Note: Results vary by hardware and model*

### Feature Comparison

| Feature | vLLM | TorchServe | Triton |
|---------|------|------------|--------|
| GPU Support | ✅ | ✅ | ✅ |
| CPU Support | ✅ (via fallback) | ✅ | ✅ |
| Auto-start | - | ✅ | ✅ |
| Model Versioning | ❌ | ✅ | ✅ |
| Multi-model | ❌ | ✅ | ✅ |
| Batch Inference | ✅ | ✅ | ✅ |
| Streaming | ✅ | ❌ | ❌ |

## Troubleshooting

### vLLM Issues

**Problem**: GPU hang on initialization

```
[EngineCore_DP0] Initializing a V1 LLM engine...
(hangs indefinitely)
```

**Solution**: Use eager mode
```python
llm = LLM(model="gpt2", enforce_eager=True)
```

**Problem**: GPU memory error

**Solution**: Reduce memory utilization
```python
llm = LLM(model="gpt2", gpu_memory_utilization=0.15)
```

### TorchServe Issues

**Problem**: Server start timeout

```
Failed to start TorchServe: timeout after 60 seconds
```

**Solution**: Automatically falls back to transformers. Check logs:
```bash
cat ~/.rm_abstract/torchserve.log
```

**Problem**: Port already in use

**Solution**: Stop existing TorchServe
```bash
torchserve --stop
```

### Triton Issues

**Problem**: Docker not available

```
Triton backend not available: Docker not found
```

**Solution**: Install and start Docker
```bash
docker --version  # Check installation
sudo systemctl start docker  # Start daemon
```

**Problem**: ONNX conversion failed

```
Missing dependency for ONNX conversion: No module named 'onnxscript'
```

**Solution**: Install ONNX tools
```bash
pip install onnx onnxscript
```

## Best Practices

### 1. Development vs Production

**Development**:
```python
# Fast iteration with vLLM + CPU fallback
llm = LLM(model="gpt2", device="auto", engine="vllm")
```

**Production**:
```python
# Reliable serving with TorchServe
llm = LLM(model="gpt2", device="cpu", engine="torchserve")
```

### 2. GPU Selection

Always specify GPU to avoid conflicts:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

llm = LLM(model="gpt2", device="nvidia")
```

### 3. Resource Cleanup

Engines auto-cleanup on exit, but you can manually cleanup:

```python
# Context manager (recommended)
from rm_abstract import LLM

with LLM(model="gpt2", engine="torchserve") as llm:
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
    # Auto cleanup

# Manual cleanup
llm = LLM(model="gpt2", engine="triton")
try:
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=10))
finally:
    llm.cleanup()  # If implemented
```

### 4. Engine Fallbacks

RM Abstract automatically falls back to transformers if an engine fails:

```python
# If Triton fails, automatically uses transformers
llm = LLM(model="gpt2", engine="triton")
# Logs: "Triton backend not available, falling back to transformers"
```

## Examples

See the `examples/` directory:

- `01_quick_start.py` - Simplest usage
- `02_device_selection.py` - Device control
- `03_auto_import_hook.py` - Import hooking
- `04_serving_engines.py` - Engine testing
- `05_engine_comparison.py` - Performance comparison

## Next Steps

- [Device Options](DEVICE_OPTIONS.md)
- [Auto Import Hook](AUTO_IMPORT.md)
- [Performance Tuning](PERFORMANCE.md)

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

## Example 1: Basic Usage

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
```

---

## Example 2: Device Switching

```python
import rm_abstract

# Start with CPU
rm_abstract.init(device="cpu", verbose=True)

# Check current device
print(f"Current: {rm_abstract.get_device_info()}")

# Switch to GPU (if available)
try:
    rm_abstract.switch_device("nvidia")
    print(f"Switched to: {rm_abstract.get_device_info()}")
except Exception as e:
    print(f"GPU not available: {e}")
```

---

## Example 3: Plugin Registry

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

## Example 4: Hardware Detection

```python
from rm_abstract import HardwareDetector

detector = HardwareDetector()

# Print summary
print(detector.summary())

# Check specific vendors
vendors = ["nvidia", "rebellions", "furiosa", "cpu"]
for vendor in vendors:
    available = detector.is_available(vendor)
    print(f"  {vendor}: {'Yes' if available else 'No'}")

# Get best available
best = detector.get_best_available()
if best:
    print(f"Best: {best.hardware_type.value} - {best.name}")
```

---

## Example 5: Serving Engines

```python
from rm_abstract.engines import (
    VLLMEngine,
    ServingConfig,
    ServingEngineType,
    DeviceTarget,
)

# Check if vLLM is available
if VLLMEngine.is_available():
    config = ServingConfig(
        engine=ServingEngineType.VLLM,
        device=DeviceTarget.GPU,
        model_name="gpt2",
    )
    engine = VLLMEngine(config)
    print(f"Engine: {engine.name}")
    print(f"Supported devices: {engine.supported_devices()}")
else:
    print("vLLM not available")
```

---

## Device Options

```python
import rm_abstract

# Auto-select (NPU > GPU > CPU)
rm_abstract.init(device="auto")

# Specific devices
rm_abstract.init(device="nvidia:0")  # NVIDIA GPU
rm_abstract.init(device="gpu:0")     # Alias for nvidia
rm_abstract.init(device="cpu")       # CPU
rm_abstract.init(device="rbln:0")    # Rebellions NPU
rm_abstract.init(device="furiosa:0") # FuriosaAI NPU
```

---

## Error Handling

```python
from rm_abstract.exceptions import (
    RMAbstractError,
    BackendNotAvailableError,
    NotInitializedError,
)

try:
    rm_abstract.init(device="gpu:0")
except BackendNotAvailableError:
    print("GPU not available, falling back to CPU")
    rm_abstract.init(device="cpu")
```

---

## Running Examples

```bash
# Basic usage
python examples/basic_usage.py

# Device switching
python examples/device_switching.py

# Hardware detection
python examples/hardware_detection.py

# Plugin registry
python examples/plugin_registry.py

# Serving engines
python examples/serving_engine.py
```

---

## Next Steps

- [README.md](README.md) - Full documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

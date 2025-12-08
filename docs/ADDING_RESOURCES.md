# Adding New Resources Guide

This guide explains how to add new hardware resources (GPU, NPU, etc.) to RM Abstract Layer.

---

## Overview

The plugin architecture allows you to add new resources without modifying core code. Each resource:

1. Implements the `Resource` abstract base class
2. Registers with the `PluginRegistry`
3. Is auto-discovered at runtime

---

## Step 1: Create Resource Class

Create a new file in `src/rm_abstract/resources/`:

```text
src/rm_abstract/resources/
├── base.py              # Base classes (don't modify)
├── gpu/
│   └── nvidia.py        # Example: NVIDIA GPU
├── npu/
│   ├── rebellions.py    # Example: Rebellions ATOM
│   └── furiosa.py       # Example: FuriosaAI RNGD
├── cpu/
│   └── default.py       # Example: CPU fallback
└── your_vendor/         # <- Create new folder
    ├── __init__.py
    └── your_resource.py # <- Create new resource
```

### Resource Template

```python
# src/rm_abstract/resources/your_vendor/your_resource.py

"""
YourVendor Resource Implementation

Supports: YourVendor Hardware Accelerator
SDK: your-sdk >= 1.0.0
"""

from typing import Any, Dict, Optional
import logging

from ..base import Resource, ResourceType, ResourceInfo, ResourceCapability

logger = logging.getLogger(__name__)


class YourResource(Resource):
    """
    YourVendor hardware resource.

    Uses your-sdk for inference acceleration.
    """

    # Class-level constants
    RESOURCE_TYPE = ResourceType.NPU  # or GPU, CPU
    VENDOR = "your_vendor"
    NAME = "YourResource"

    def __init__(self, device_id: int = 0, **kwargs):
        """
        Initialize resource.

        Args:
            device_id: Device index (default: 0)
            **kwargs: Additional options
        """
        super().__init__(device_id, **kwargs)
        self._model = None
        self._engine = None

    @staticmethod
    def is_available() -> bool:
        """
        Check if resource is available.

        This is called as a static method by the plugin registry,
        so it must not rely on instance state.

        Returns:
            True if SDK is installed and hardware is detected
        """
        try:
            # 1. Check SDK installation
            import your_sdk

            # 2. Check hardware availability
            devices = your_sdk.list_devices()
            return len(devices) > 0

        except ImportError:
            logger.debug("your-sdk not installed")
            return False
        except Exception as e:
            logger.debug(f"Hardware not available: {e}")
            return False

    def initialize(self) -> None:
        """Initialize the resource."""
        if self._initialized:
            return

        try:
            import your_sdk

            # Initialize SDK/runtime
            self._engine = your_sdk.create_engine(device_id=self.device_id)
            self._initialized = True

            logger.info(f"Initialized {self.name} on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def prepare_model(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Prepare model for this resource.

        This may involve:
        - Model conversion (e.g., to ONNX)
        - Compilation for the hardware
        - Optimization

        Args:
            model: Original model (PyTorch, HuggingFace, etc.)
            config: Optional configuration

        Returns:
            Prepared/compiled model
        """
        if not self._initialized:
            self.initialize()

        config = config or {}

        try:
            import your_sdk

            # Example: Compile model for hardware
            compiled = your_sdk.compile(
                model,
                device_id=self.device_id,
                **config
            )

            self._model = compiled
            return compiled

        except Exception as e:
            logger.error(f"Model preparation failed: {e}")
            raise

    def execute(
        self,
        model: Any,
        inputs: Any,
        **kwargs
    ) -> Any:
        """
        Execute inference.

        Args:
            model: Prepared model
            inputs: Input data
            **kwargs: Additional inference options

        Returns:
            Inference output
        """
        if not self._initialized:
            raise RuntimeError("Resource not initialized")

        try:
            # Run inference
            outputs = model.run(inputs, **kwargs)
            return outputs

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def get_info(self) -> ResourceInfo:
        """Get resource information."""
        try:
            import your_sdk

            device = your_sdk.get_device_info(self.device_id)

            return ResourceInfo(
                resource_type=self.RESOURCE_TYPE,
                device_id=self.device_id,
                name=device.name,
                vendor=self.VENDOR,
                memory_total=device.memory_total,
                memory_available=device.memory_available,
                extra={
                    "sdk_version": your_sdk.__version__,
                    "driver_version": device.driver_version,
                },
            )
        except Exception:
            return ResourceInfo(
                resource_type=self.RESOURCE_TYPE,
                device_id=self.device_id,
                name=self.NAME,
                vendor=self.VENDOR,
            )

    def get_capability(self) -> ResourceCapability:
        """Get resource capabilities."""
        return ResourceCapability(
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_dynamic_batching=True,
            max_batch_size=32,
        )

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._engine is not None:
            try:
                self._engine.close()
            except Exception:
                pass
            self._engine = None

        self._model = None
        super().cleanup()
```

### Create `__init__.py`

```python
# src/rm_abstract/resources/your_vendor/__init__.py

from .your_resource import YourResource

__all__ = ["YourResource"]
```

---

## Step 2: Register Plugin

Add your resource to the discovery module:

```python
# src/rm_abstract/registry/discovery.py

def discover_resources(registry: PluginRegistry) -> None:
    """Discover and register all available resources."""

    # ... existing resources ...

    # YourVendor Resource
    try:
        from ..resources.your_vendor.your_resource import YourResource

        registry.register(PluginInfo(
            name="your_vendor",           # Device name: rm_abstract.init(device="your_vendor:0")
            plugin_type=PluginType.RESOURCE,
            cls=YourResource,
            vendor="your_vendor",
            description="YourVendor hardware accelerator",
            priority=100,                 # NPU=100, GPU=50, CPU=0
            aliases=["yv", "your"],       # Optional aliases
        ))
        logger.debug("Registered your_vendor resource")

    except ImportError as e:
        logger.debug(f"your_vendor resource not available: {e}")
```

---

## Step 3: Add to Resources `__init__.py`

```python
# src/rm_abstract/resources/__init__.py

# ... existing exports ...

# YourVendor
try:
    from .your_vendor import YourResource
    __all__.append("YourResource")
except ImportError:
    pass
```

---

## Step 4: Write Tests

```python
# tests/unit/test_your_resource.py

"""Tests for YourResource"""

import pytest
from unittest.mock import patch, MagicMock


class TestYourResource:
    """YourResource tests"""

    def test_is_available_without_sdk(self):
        """Test is_available returns False when SDK not installed"""
        with patch.dict("sys.modules", {"your_sdk": None}):
            from rm_abstract.resources.your_vendor import YourResource
            assert YourResource.is_available() is False

    def test_is_available_with_sdk(self):
        """Test is_available with mocked SDK"""
        mock_sdk = MagicMock()
        mock_sdk.list_devices.return_value = [{"id": 0}]

        with patch.dict("sys.modules", {"your_sdk": mock_sdk}):
            from rm_abstract.resources.your_vendor import YourResource
            # Force reimport
            assert YourResource.is_available() is True

    def test_resource_creation(self):
        """Test resource can be created"""
        from rm_abstract.resources.your_vendor import YourResource

        resource = YourResource(device_id=0)
        assert resource.device_id == 0
        assert resource.name == "YourResource"
        assert resource.VENDOR == "your_vendor"

    def test_get_info(self):
        """Test get_info returns valid ResourceInfo"""
        from rm_abstract.resources.your_vendor import YourResource
        from rm_abstract.resources.base import ResourceType

        resource = YourResource(device_id=0)
        info = resource.get_info()

        assert info.resource_type == ResourceType.NPU
        assert info.device_id == 0
```

---

## Priority Guidelines

| Priority | Resource Type | Use Case |
|----------|---------------|----------|
| 100+ | NPU (specialized) | Rebellions, FuriosaAI, etc. |
| 50-99 | GPU | NVIDIA, AMD |
| 1-49 | Accelerator | FPGA, TPU |
| 0 | CPU | Fallback |

Higher priority resources are selected first in `auto` mode.

---

## Checklist

- [ ] Resource class implements all abstract methods
- [ ] `is_available()` is a `@staticmethod`
- [ ] Plugin registered in `discovery.py`
- [ ] Added to `resources/__init__.py`
- [ ] Unit tests written
- [ ] Documentation updated

---

## Example Usage

After adding your resource:

```python
import rm_abstract

# Check availability
resources = rm_abstract.get_available_resources()
print(resources)  # {'nvidia': False, 'your_vendor': True, 'cpu': True}

# Initialize with your resource
rm_abstract.init(device="your_vendor:0")

# Or use auto-select (will pick highest priority available)
rm_abstract.init(device="auto")

# Switch to your resource
rm_abstract.switch_device("your_vendor")
```

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guide

# Adding New Serving Engines Guide

This guide explains how to add new serving engines to RM Abstract Layer.

---

## Overview

Serving engines handle model deployment and inference serving. Each engine:

1. Implements the `ServingEngine` abstract base class
2. Registers with the `PluginRegistry`
3. Provides unified interface for model loading and inference

---

## Step 1: Create Engine Class

Create a new file in `src/rm_abstract/engines/`:

```text
src/rm_abstract/engines/
├── base.py           # Base classes (don't modify)
├── vllm.py           # Example: vLLM
├── triton.py         # Example: Triton
├── torchserve.py     # Example: TorchServe
└── your_engine.py    # <- Create new engine
```

### Engine Template

```python
# src/rm_abstract/engines/your_engine.py

"""
YourEngine Serving Engine Implementation

Supports: Your serving framework
"""

from typing import Any, Dict, List, Optional
import logging

from .base import ServingEngine, ServingConfig, DeviceTarget

logger = logging.getLogger(__name__)


class YourEngine(ServingEngine):
    """
    Your serving engine implementation.

    Features:
    - Feature 1
    - Feature 2
    """

    def __init__(self, config: ServingConfig):
        """
        Initialize engine.

        Args:
            config: Serving configuration
        """
        super().__init__(config)
        self._server = None
        self._models: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Engine name."""
        return "YourEngine"

    @staticmethod
    def is_available() -> bool:
        """
        Check if engine is available.

        Returns:
            True if required packages are installed
        """
        try:
            import your_serving_lib
            return True
        except ImportError:
            return False

    def supported_devices(self) -> List[DeviceTarget]:
        """
        Get supported device targets.

        Returns:
            List of supported devices
        """
        return [DeviceTarget.GPU, DeviceTarget.CPU]

    def start(self) -> None:
        """Start the serving engine."""
        if self._running:
            return

        try:
            import your_serving_lib

            self._server = your_serving_lib.Server(
                host=self.config.host,
                port=self.config.port,
            )
            self._server.start()
            self._running = True

            logger.info(f"Started {self.name} on port {self.config.port}")

        except Exception as e:
            logger.error(f"Failed to start: {e}")
            raise

    def stop(self) -> None:
        """Stop the serving engine."""
        if not self._running:
            return

        try:
            if self._server:
                self._server.stop()
                self._server = None
            self._running = False

            logger.info(f"Stopped {self.name}")

        except Exception as e:
            logger.error(f"Failed to stop: {e}")

    def load_model(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Load a model.

        Args:
            model_name: Model identifier
            model_path: Optional path to model
            **kwargs: Additional options

        Returns:
            Loaded model handle
        """
        if not self._running:
            self.start()

        try:
            import your_serving_lib

            model = your_serving_lib.load_model(
                model_name,
                path=model_path,
                **kwargs
            )
            self._models[model_name] = model

            logger.info(f"Loaded model: {model_name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model.

        Args:
            model_name: Model identifier
        """
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"Unloaded model: {model_name}")

    def infer(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Run inference.

        Args:
            prompt: Input prompt
            model_name: Model to use (default: config.model_name)
            **kwargs: Inference options

        Returns:
            Generated text
        """
        model_name = model_name or self.config.model_name

        if model_name not in self._models:
            self.load_model(model_name)

        model = self._models[model_name]

        try:
            output = model.generate(prompt, **kwargs)
            return output

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check engine health.

        Returns:
            True if engine is healthy
        """
        if not self._running:
            return False

        try:
            # Implement health check logic
            return self._server.is_healthy()
        except Exception:
            return False
```

---

## Step 2: Register Plugin

Add your engine to the discovery module:

```python
# src/rm_abstract/registry/discovery.py

def discover_engines(registry: PluginRegistry) -> None:
    """Discover and register all available engines."""

    # ... existing engines ...

    # YourEngine
    try:
        from ..engines.your_engine import YourEngine

        registry.register(PluginInfo(
            name="your_engine",
            plugin_type=PluginType.ENGINE,
            cls=YourEngine,
            vendor="your_vendor",
            description="Your serving engine",
            priority=50,  # Adjust based on preference
        ))
        logger.debug("Registered your_engine")

    except ImportError as e:
        logger.debug(f"your_engine not available: {e}")
```

---

## Step 3: Add to Engines `__init__.py`

```python
# src/rm_abstract/engines/__init__.py

from .base import ServingEngine, ServingConfig, ServingEngineType, DeviceTarget
from .vllm import VLLMEngine
from .triton import TritonEngine
from .torchserve import TorchServeEngine

# Add your engine
try:
    from .your_engine import YourEngine
except ImportError:
    YourEngine = None

__all__ = [
    "ServingEngine",
    "ServingConfig",
    "ServingEngineType",
    "DeviceTarget",
    "VLLMEngine",
    "TritonEngine",
    "TorchServeEngine",
    "YourEngine",
]
```

---

## Step 4: Add Engine Type (Optional)

If you want to support the engine via `ServingEngineType` enum:

```python
# src/rm_abstract/engines/base.py

class ServingEngineType(Enum):
    VLLM = "vllm"
    TRITON = "triton"
    TORCHSERVE = "torchserve"
    YOUR_ENGINE = "your_engine"  # Add this
```

---

## Step 5: Write Tests

```python
# tests/unit/test_your_engine.py

"""Tests for YourEngine"""

import pytest
from unittest.mock import patch, MagicMock


class TestYourEngine:
    """YourEngine tests"""

    def test_is_available_without_lib(self):
        """Test is_available returns False when lib not installed"""
        with patch.dict("sys.modules", {"your_serving_lib": None}):
            from rm_abstract.engines.your_engine import YourEngine
            assert YourEngine.is_available() is False

    def test_engine_name(self):
        """Test engine name property"""
        from rm_abstract.engines import YourEngine, ServingConfig, ServingEngineType

        config = ServingConfig(engine=ServingEngineType.YOUR_ENGINE)
        engine = YourEngine(config)

        assert engine.name == "YourEngine"

    def test_supported_devices(self):
        """Test supported_devices returns valid list"""
        from rm_abstract.engines import YourEngine, ServingConfig, DeviceTarget

        config = ServingConfig()
        engine = YourEngine(config)

        devices = engine.supported_devices()
        assert DeviceTarget.GPU in devices or DeviceTarget.CPU in devices
```

---

## Checklist

- [ ] Engine class implements all abstract methods
- [ ] `is_available()` is a `@staticmethod`
- [ ] Plugin registered in `discovery.py`
- [ ] Added to `engines/__init__.py`
- [ ] Unit tests written
- [ ] Documentation updated

---

## Example Usage

After adding your engine:

```python
from rm_abstract.engines import (
    YourEngine,
    ServingConfig,
    ServingEngineType,
)

# Check availability
if YourEngine.is_available():
    config = ServingConfig(
        engine=ServingEngineType.YOUR_ENGINE,
        model_name="gpt2",
        port=8000,
    )

    engine = YourEngine(config)
    engine.start()
    engine.load_model("gpt2")

    output = engine.infer("Hello, world!")
    print(output)

    engine.stop()
```

---

## Related Documents

- [ADDING_RESOURCES.md](ADDING_RESOURCES.md) - Adding hardware resources
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture

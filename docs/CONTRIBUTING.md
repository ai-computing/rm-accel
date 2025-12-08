# Contributing Guide

How to contribute to RM Abstract Layer.

---

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/rm_abstract_layer.git
cd rm_abstract_layer
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
```

### 3. Install Dependencies

```bash
pip install -e ".[dev,all]"
```

### 4. Verify Installation

```bash
pytest tests/ -v
```

---

## Project Structure

```text
rm_abstract_layer/
├── src/rm_abstract/      # Source code
│   ├── common/           # Shared utilities
│   ├── resources/        # Hardware resources
│   ├── engines/          # Serving engines
│   ├── registry/         # Plugin system
│   └── core/             # Core modules
├── tests/                # Tests
├── examples/             # Examples
└── requirements/         # Dependencies
```

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific file
pytest tests/test_core.py -v
pytest tests/unit/test_registry.py -v

# With coverage
pytest tests/ --cov=src/rm_abstract --cov-report=html
```

### Write Tests

```python
# tests/test_example.py

import pytest

class TestExample:
    def test_something(self):
        """Test description"""
        result = some_function()
        assert result == expected
```

---

## Adding New Resources

### 1. Create Resource Class

```python
# src/rm_abstract/resources/new_vendor/new_resource.py

from ..base import Resource, ResourceType

class NewResource(Resource):
    RESOURCE_TYPE = ResourceType.NPU
    VENDOR = "new_vendor"
    NAME = "NewResource"

    @staticmethod
    def is_available() -> bool:
        try:
            import new_sdk
            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        self._initialized = True

    def prepare_model(self, model, config=None):
        return model

    def execute(self, model, inputs, **kwargs):
        return outputs

    def cleanup(self) -> None:
        self._initialized = False
```

### 2. Register Plugin

```python
# src/rm_abstract/registry/discovery.py

def discover_resources(registry: PluginRegistry) -> None:
    # Add new resource
    try:
        from ..resources.new_vendor.new_resource import NewResource
        registry.register(PluginInfo(
            name="new",
            plugin_type=PluginType.RESOURCE,
            cls=NewResource,
            vendor="new_vendor",
            priority=100,
        ))
    except ImportError:
        pass
```

### 3. Write Tests

```python
# tests/unit/test_new_resource.py

class TestNewResource:
    def test_is_available(self):
        from rm_abstract.resources.new_vendor import NewResource
        # Test availability
```

---

## Code Style

### Formatting

```bash
# Black formatting
black src/ tests/

# isort import sorting
isort src/ tests/
```

### Linting

```bash
# flake8
flake8 src/ tests/

# mypy type check
mypy src/
```

---

## Pull Request Guide

### Branch Naming

```text
feature/feature-name
fix/bug-name
docs/doc-name
refactor/refactor-name
```

### Commit Messages

```text
feat: Add new feature
fix: Fix bug
docs: Update documentation
refactor: Refactor code
test: Add/update tests
chore: Other tasks
```

### PR Checklist

- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code formatted (black, isort)
- [ ] Lint passes (flake8, mypy)

---

## Questions

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

## License

Contributions are under MIT License.

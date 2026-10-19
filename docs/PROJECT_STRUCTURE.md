# Project Structure

## Directory Layout

```
rm_abstract_layer/
├── docs/                          # 📚 Documentation
│   ├── QUICKSTART.md              # Quick start guide
│   ├── SERVING_ENGINES.md         # Serving engines guide
│   ├── README_REQUIREMENTS.md     # Requirements installation
│   ├── ARCHITECTURE.md            # System architecture
│   ├── CONTRIBUTING.md            # Development guide
│   ├── ADDING_RESOURCES.md        # Adding new hardware
│   ├── ADDING_ENGINES.md          # Adding new engines
│   └── archive/                   # 0.2–0.3 test reports and slides (not maintained)
│
├── examples/                      # 🎯 Examples (cleaned up!)
│   ├── 01_quick_start.py          # Simplest usage
│   ├── 02_device_selection.py     # Device control
│   ├── 03_auto_import_hook.py     # Auto import hooking
│   ├── 04_serving_engines.py      # Test all engines
│   └── 05_engine_comparison.py    # Performance comparison
│
├── requirements/                  # 📦 Dependencies (organized!)
│   ├── requirements.txt           # Base (CPU)
│   ├── requirements-gpu.txt       # GPU support (vLLM)
│   ├── requirements-triton.txt    # Triton engine
│   ├── requirements-torchserve.txt# TorchServe engine
│   ├── requirements-serving.txt   # All engines
│   ├── requirements-dev.txt       # Development tools
│   └── requirements-all.txt       # Everything
│
├── src/rm_abstract/               # 🔧 Source code
│   ├── __init__.py                # Main API exports
│   ├── airm/                      # ai-rm protocol v0.1: AcceleratorProvider, node daemon
│   ├── auto.py                    # Auto import hooking ⭐
│   ├── llm.py                     # LLM proxy class
│   ├── exceptions.py              # Exception classes
│   │
│   ├── compat/                    # Compatibility layer
│   │   ├── vllm_compat.py         # vLLM compatibility
│   │   └── transformers_compat.py # transformers compatibility
│   │
│   ├── common/                    # Shared utilities
│   │   ├── input_adapter.py       # Input format handling
│   │   └── output_adapter.py      # Output format conversion
│   │
│   ├── resources/                 # Hardware-specific
│   │   ├── base.py                # Resource ABC
│   │   ├── gpu/nvidia.py          # NVIDIA GPU (vLLM)
│   │   ├── npu/rebellions.py      # Rebellions ATOM NPU
│   │   ├── npu/furiosa.py         # FuriosaAI RNGD NPU
│   │   └── cpu/default.py         # CPU fallback
│   │
│   ├── engines/                   # Serving engines ⭐
│   │   ├── base.py                # ServingEngine base
│   │   ├── vllm.py                # vLLM engine
│   │   ├── triton.py              # Triton engine ✅
│   │   ├── torchserve.py          # TorchServe engine ✅
│   │   └── model_converter.py     # Model conversion
│   │
│   ├── registry/                  # Plugin system
│   │   ├── plugin_registry.py     # Centralized registry
│   │   └── discovery.py           # Auto-discovery
│   │
│   └── core/                      # Core modules
│       ├── config.py              # Configuration
│       └── controller.py          # DeviceFlowController
│
├── tests/                         # 🧪 Test suite
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── conftest.py                # Pytest configuration
│
├── deploy/helm/rm-abstract-node/  # Helm chart for the node daemon (DaemonSet)
├── scripts/                       # setup scripts, run_examples.sh, manual/multi_engine_check.py
├── tools/check_sensitive.py       # internal addresses / tokens / keys scan
├── ai-rm.yaml                     # ai-rm component manifest
├── CHANGELOG.md
├── pyproject.toml                 # Project metadata
└── README.md                      # Main README
```

## Key Files

### Entry Points

| File | Purpose |
|------|---------|
| `src/rm_abstract/__init__.py` | Main API (`LLM`, `SamplingParams`, `init`, etc.) |
| `src/rm_abstract/auto.py` | Auto import hooking (`use()`, hooks) |
| `src/rm_abstract/llm.py` | LLM proxy class (device routing) |

### Serving Engines

| File | Status | Description |
|------|--------|-------------|
| `src/rm_abstract/engines/vllm.py` | ✅ Working | vLLM engine |
| `src/rm_abstract/engines/torchserve.py` | ✅ Working | TorchServe with auto-start |
| `src/rm_abstract/engines/triton.py` | ✅ Working | Triton with Docker auto-start |

### Compatibility Layer

| File | Purpose |
|------|---------|
| `src/rm_abstract/compat/vllm_compat.py` | vLLM API compatibility |
| `src/rm_abstract/compat/transformers_compat.py` | transformers fallback |

### Documentation

| File | Audience |
|------|----------|
| `docs/QUICKSTART.md` | New users (5-min guide) |
| `docs/SERVING_ENGINES.md` | All users (engine details) |
| `docs/README_REQUIREMENTS.md` | All users (installation) |
| `docs/archive/FINAL_TEST_REPORT.md` | All users (test results) |
| `docs/ARCHITECTURE.md` | Developers |
| `docs/CONTRIBUTING.md` | Contributors |

## File Count

- **Examples**: 5 (cleaned from 16)
- **Requirements**: 7 (organized in folder)
- **Documentation**: 9 (organized in docs/)
- **Source files**: ~20 core modules
- **Tests**: ~10 test files

## Changes Made

### ✅ Cleaned Up

1. **Examples**: Removed 11 duplicate/legacy examples
2. **Requirements**: Organized into `requirements/` folder
3. **Documentation**: Organized into `docs/` folder

### ✅ Added

1. **New Examples**: 5 comprehensive examples (01-05)
2. **New Docs**:
   - `QUICKSTART.md`
   - `SERVING_ENGINES.md`
   - `README_REQUIREMENTS.md`
   - `archive/FINAL_TEST_REPORT.md`
   - `PROJECT_STRUCTURE.md` (this file)

### ✅ Improved

1. **Auto-hook**: GPU/NPU auto-disable hooks for performance
2. **Engine fallbacks**: All engines fall back to transformers
3. **Requirements**: Modular installation options

## Quick Navigation

### For Users

1. **Getting Started**: `docs/QUICKSTART.md`
2. **Installation**: `docs/README_REQUIREMENTS.md`
3. **Examples**: `examples/01_quick_start.py`
4. **Engine Guide**: `docs/SERVING_ENGINES.md`

### For Developers

1. **Architecture**: `docs/ARCHITECTURE.md`
2. **Contributing**: `docs/CONTRIBUTING.md`
3. **Adding Resources**: `docs/ADDING_RESOURCES.md`
4. **Adding Engines**: `docs/ADDING_ENGINES.md`

### For Testers

1. **Test Report**: `docs/archive/FINAL_TEST_REPORT.md`
2. **Test Script**: `scripts/manual/multi_engine_check.py`
3. **Engine Tests**: `examples/04_serving_engines.py`

## Next Steps

See `docs/archive/FINAL_TEST_REPORT.md` for:
- Test results (all engines ✅)
- Known issues
- Recommendations

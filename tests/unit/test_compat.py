"""Tests for compatibility layer."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_values(self):
        """Test default sampling parameters."""
        from rm_abstract.compat.vllm_compat import SamplingParams

        params = SamplingParams()

        assert params.n == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.max_tokens == 16

    def test_custom_values(self):
        """Test custom sampling parameters."""
        from rm_abstract.compat.vllm_compat import SamplingParams

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
        )

        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.max_tokens == 100
        assert params.stop == ["END"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from rm_abstract.compat.vllm_compat import SamplingParams

        params = SamplingParams(temperature=0.5, max_tokens=50)
        d = params.to_dict()

        assert isinstance(d, dict)
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 50


class TestRequestOutput:
    """Tests for RequestOutput."""

    def test_request_output_creation(self):
        """Test creating RequestOutput."""
        from rm_abstract.compat.vllm_compat import RequestOutput, CompletionOutput

        output = RequestOutput(
            request_id="0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="World")],
        )

        assert output.request_id == "0"
        assert output.prompt == "Hello"
        assert output.output.text == "World"


class TestLLMDeviceSelection:
    """Tests for LLM device selection."""

    def test_resolve_device_explicit(self):
        """Test explicit device selection."""
        from rm_abstract.compat.vllm_compat import LLM

        # Create LLM with explicit device
        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu")
            assert llm.device == "cpu"

    def test_resolve_device_from_env(self):
        """Test device selection from environment."""
        from rm_abstract.compat.vllm_compat import LLM

        with patch.dict(os.environ, {"RM_DEVICE": "rbln"}):
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2")
                assert llm.device == "rbln"

    def test_resolve_device_with_index(self):
        """Test device with index (e.g., nvidia:0)."""
        from rm_abstract.compat.vllm_compat import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="nvidia:0")
            resolved = llm._resolve_device("nvidia:0")
            assert resolved == "nvidia"


class TestAutoModelDeviceSelection:
    """Tests for AutoModelForCausalLM device selection."""

    def test_device_from_env(self):
        """Test device selection from environment."""
        with patch.dict(os.environ, {"RM_DEVICE": "cpu"}):
            from rm_abstract.compat.transformers_compat import _resolve_device
            assert _resolve_device("auto") == "cpu"

    def test_explicit_device(self):
        """Test explicit device selection."""
        from rm_abstract.compat.transformers_compat import _resolve_device
        assert _resolve_device("rbln") == "rbln"
        assert _resolve_device("nvidia:0") == "nvidia"


class TestAutoHook:
    """Tests for auto-hook functionality."""

    def test_enable_disable_hooks(self):
        """Test enabling and disabling hooks."""
        from rm_abstract.auto import enable_hooks, disable_hooks, is_hooked

        # Ensure hooks are enabled
        enable_hooks()
        assert is_hooked()

        # Disable hooks
        disable_hooks()
        assert not is_hooked()

        # Re-enable for other tests
        enable_hooks()

    def test_hook_finder_installed(self):
        """Test that hook finder is in sys.meta_path."""
        from rm_abstract.auto import RMAbstractFinder, enable_hooks

        enable_hooks()

        found = False
        for finder in sys.meta_path:
            if isinstance(finder, RMAbstractFinder):
                found = True
                break

        assert found

    def test_use_function(self):
        """Test use() function for explicit configuration."""
        from rm_abstract.auto import use, get_config

        # Set device and engine
        use(device="cpu", engine="vllm")
        config = get_config()

        assert config["device"] == "cpu"
        assert config["engine"] == "vllm"

    def test_use_sets_env_vars(self):
        """Test that use() sets environment variables."""
        from rm_abstract.auto import use

        use(device="rbln", engine="triton")

        assert os.environ.get("RM_DEVICE") == "rbln"
        assert os.environ.get("RM_ENGINE") == "triton"

        # Reset
        use(device="auto", engine="vllm")

    def test_get_config(self):
        """Test get_config returns current configuration."""
        from rm_abstract.auto import use, get_config

        use(device="nvidia", engine="vllm")
        config = get_config()

        assert isinstance(config, dict)
        assert "device" in config
        assert "engine" in config
        assert config["device"] == "nvidia"


class TestCompat:
    """Tests for compat module exports."""

    def test_vllm_compat_exports(self):
        """Test vllm_compat exports."""
        from rm_abstract.compat import vllm_compat

        assert hasattr(vllm_compat, "LLM")
        assert hasattr(vllm_compat, "SamplingParams")
        assert hasattr(vllm_compat, "RequestOutput")

    def test_transformers_compat_exports(self):
        """Test transformers_compat exports."""
        from rm_abstract.compat import transformers_compat

        assert hasattr(transformers_compat, "AutoModelForCausalLM")
        assert hasattr(transformers_compat, "AutoTokenizer")

    def test_compat_init_exports(self):
        """Test compat __init__ exports."""
        from rm_abstract.compat import (
            LLM,
            SamplingParams,
            RequestOutput,
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        assert LLM is not None
        assert SamplingParams is not None
        assert RequestOutput is not None
        assert AutoModelForCausalLM is not None
        assert AutoTokenizer is not None

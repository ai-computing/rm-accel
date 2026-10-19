"""Tests for compatibility layer (legacy compat module)."""

import os
import pytest
from unittest.mock import patch, MagicMock


class TestCompatSamplingParams:
    """Tests for compat.SamplingParams (legacy)."""

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


class TestCompatRequestOutput:
    """Tests for compat.RequestOutput (legacy)."""

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


class TestCompatLLMDeviceSelection:
    """Tests for compat.LLM device selection (legacy)."""

    def test_resolve_device_explicit(self):
        """Test explicit device selection."""
        from rm_abstract.compat.vllm_compat import LLM

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


class TestCompatTransformers:
    """Tests for transformers_compat (legacy)."""

    def test_device_from_env(self):
        """Test device selection from environment."""
        from rm_abstract.compat import transformers_compat as tc

        # RM_DEVICE wins over host detection, so the result must not depend on the host.
        with patch.dict(os.environ, {"RM_DEVICE": "cpu"}), patch.object(
            tc, "_detect_device", side_effect=AssertionError("must not auto-detect")
        ):
            assert tc._resolve_device(tc._get_device()) == "cpu"

    def test_auto_device_uses_detection(self):
        """Test 'auto' delegates to host detection."""
        from rm_abstract.compat import transformers_compat as tc

        with patch.object(tc, "_detect_device", return_value="furiosa"):
            assert tc._resolve_device("auto") == "furiosa"

    def test_explicit_device(self):
        """Test explicit device selection."""
        from rm_abstract.compat.transformers_compat import _resolve_device
        assert _resolve_device("rbln") == "rbln"
        assert _resolve_device("nvidia:0") == "nvidia"


class TestCompatModuleExports:
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


class TestCompatEngineSelection:
    """Tests for engine selection in compat layer (legacy)."""

    def test_llm_engine_parameter(self):
        """Test LLM accepts engine parameter."""
        from rm_abstract.compat.vllm_compat import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu", engine="vllm")
            assert llm.engine == "vllm"

            llm = LLM(model="gpt2", device="cpu", engine="triton")
            assert llm.engine == "triton"

            llm = LLM(model="gpt2", device="cpu", engine="torchserve")
            assert llm.engine == "torchserve"

    def test_llm_engine_from_env(self):
        """Test LLM reads engine from environment."""
        from rm_abstract.compat.vllm_compat import LLM

        with patch.dict(os.environ, {"RM_ENGINE": "triton"}):
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2", device="cpu")
                assert llm.engine == "triton"

    def test_engine_type_property(self):
        """Test engine_type property."""
        from rm_abstract.compat.vllm_compat import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu", engine="triton")
            assert llm.engine_type == "triton"

    def test_is_engine_available(self):
        """Test _is_engine_available function."""
        from rm_abstract.compat.vllm_compat import _is_engine_available

        result = _is_engine_available("vllm")
        assert isinstance(result, bool)

        result = _is_engine_available("triton")
        assert isinstance(result, bool)

        result = _is_engine_available("torchserve")
        assert isinstance(result, bool)

        result = _is_engine_available("unknown")
        assert result is False

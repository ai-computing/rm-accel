"""Tests for the new LLM proxy class."""

import os
from unittest.mock import patch


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""

    def test_default_values(self):
        """Test default sampling parameters."""
        from rm_abstract import SamplingParams

        params = SamplingParams()

        assert params.n == 1
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.max_tokens == 16

    def test_custom_values(self):
        """Test custom sampling parameters."""
        from rm_abstract import SamplingParams

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
        from rm_abstract import SamplingParams

        params = SamplingParams(temperature=0.5, max_tokens=50)
        d = params.to_dict()

        assert isinstance(d, dict)
        assert d["temperature"] == 0.5
        assert d["max_tokens"] == 50


class TestRequestOutput:
    """Tests for RequestOutput and CompletionOutput."""

    def test_request_output_creation(self):
        """Test creating RequestOutput."""
        from rm_abstract import CompletionOutput, RequestOutput

        output = RequestOutput(
            request_id="0",
            prompt="Hello",
            prompt_token_ids=[1, 2, 3],
            outputs=[CompletionOutput(index=0, text="World")],
        )

        assert output.request_id == "0"
        assert output.prompt == "Hello"
        assert output.output.text == "World"

    def test_completion_output_defaults(self):
        """Test CompletionOutput default values."""
        from rm_abstract import CompletionOutput

        output = CompletionOutput(index=0, text="test")

        assert output.token_ids == []
        assert output.cumulative_logprob is None
        assert output.finish_reason is None


class TestLLMDeviceSelection:
    """Tests for LLM device selection."""

    def test_device_explicit(self):
        """Test explicit device selection."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu")
            assert llm.device == "cpu"

    def test_device_from_env(self):
        """Test device selection from environment."""
        from rm_abstract import LLM

        with patch.dict(os.environ, {"RM_DEVICE": "rbln"}):
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2")
                assert llm.device == "rbln"

    def test_device_with_index(self):
        """Test device with index (e.g., nvidia:0)."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="nvidia:0")
            resolved = llm._resolve_device("nvidia:0")
            assert resolved == "nvidia"

    def test_device_default_auto(self):
        """Test device defaults to auto."""
        from rm_abstract import LLM

        # Temporarily clear RM_DEVICE env var
        with patch.dict(os.environ, {"RM_DEVICE": ""}, clear=False):
            os.environ.pop("RM_DEVICE", None)
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2")
                assert llm.device == "auto"


class TestLLMEngineSelection:
    """Tests for LLM engine selection."""

    def test_engine_explicit(self):
        """Test explicit engine selection."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu", engine="vllm")
            assert llm.engine == "vllm"

            llm = LLM(model="gpt2", device="cpu", engine="triton")
            assert llm.engine == "triton"

            llm = LLM(model="gpt2", device="cpu", engine="torchserve")
            assert llm.engine == "torchserve"

    def test_engine_from_env(self):
        """Test engine selection from environment."""
        from rm_abstract import LLM

        with patch.dict(os.environ, {"RM_ENGINE": "triton"}):
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2", device="cpu")
                assert llm.engine == "triton"

    def test_engine_default_vllm(self):
        """Test engine defaults to vllm."""
        from rm_abstract import LLM

        with patch.dict(os.environ, {"RM_ENGINE": ""}, clear=False):
            os.environ.pop("RM_ENGINE", None)
            with patch.object(LLM, "_initialize_backend"):
                llm = LLM(model="gpt2", device="cpu")
                assert llm.engine == "vllm"


class TestLLMProperties:
    """Tests for LLM properties."""

    def test_backend_property(self):
        """Test backend property."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu")
            llm._backend = "cpu"
            assert llm.backend == "cpu"

    def test_engine_type_property(self):
        """Test engine_type property."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu", engine="triton")
            assert llm.engine_type == "triton"

    def test_repr(self):
        """Test __repr__ method."""
        from rm_abstract import LLM

        with patch.object(LLM, "_initialize_backend"):
            llm = LLM(model="gpt2", device="cpu", engine="vllm")
            llm._backend = "cpu"
            repr_str = repr(llm)
            assert "gpt2" in repr_str
            assert "cpu" in repr_str
            assert "vllm" in repr_str


class TestLLMModuleExports:
    """Tests for module exports."""

    def test_llm_exported(self):
        """Test LLM is exported from rm_abstract."""
        from rm_abstract import LLM

        assert LLM is not None

    def test_sampling_params_exported(self):
        """Test SamplingParams is exported."""
        from rm_abstract import SamplingParams

        assert SamplingParams is not None

    def test_request_output_exported(self):
        """Test RequestOutput is exported."""
        from rm_abstract import RequestOutput

        assert RequestOutput is not None

    def test_completion_output_exported(self):
        """Test CompletionOutput is exported."""
        from rm_abstract import CompletionOutput

        assert CompletionOutput is not None

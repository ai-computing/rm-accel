"""
vLLM Serving Engine

High-performance LLM serving using vLLM.
Supports GPU (CUDA) and NPU (RBLN) backends.
"""

import logging
from typing import Any, List, Union

from .base import DeviceTarget, ServingConfig, ServingEngine, ServingEngineType

logger = logging.getLogger(__name__)


class VLLMEngine(ServingEngine):
    """
    vLLM-based serving engine.

    Supports:
    - GPU: Standard CUDA-based vLLM
    - NPU (RBLN): vLLM-RBLN for Rebellions ATOM

    Example:
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=DeviceTarget.GPU,
            model_name="meta-llama/Llama-2-7b-hf",
            port=8000,
        )
        engine = VLLMEngine(config)
        engine.load_model(config.model_name)
        output = engine.infer("Hello, how are you?")
    """

    ENGINE_TYPE = ServingEngineType.VLLM
    ENGINE_NAME = "vLLM"

    def __init__(self, config: ServingConfig):
        super().__init__(config)
        self._llm = None
        self._tokenizer = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if vLLM is available"""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """vLLM supports GPU and RBLN NPU"""
        devices = []

        # Check GPU
        try:
            import torch

            if torch.cuda.is_available():
                devices.append(DeviceTarget.GPU)
        except ImportError:
            pass

        # Check RBLN NPU
        try:
            import vllm_rbln  # noqa: F401

            devices.append(DeviceTarget.NPU_RBLN)
        except ImportError:
            try:
                import rebel  # noqa: F401

                devices.append(DeviceTarget.NPU_RBLN)
            except ImportError:
                pass

        devices.append(DeviceTarget.CPU)
        return devices

    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load model with vLLM.

        Args:
            model_name_or_path: HuggingFace model name or local path
            **kwargs: Additional vLLM options
        """
        from vllm import LLM

        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        if self.config.max_seq_len:
            llm_kwargs["max_model_len"] = self.config.max_seq_len

        # Handle RBLN NPU device
        if self.config.device == DeviceTarget.NPU_RBLN:
            try:
                import vllm_rbln  # noqa: F401

                llm_kwargs["device"] = "rbln"
            except ImportError:
                try:

                    llm_kwargs["device"] = "rbln"
                except Exception:
                    raise RuntimeError(
                        "RBLN NPU requires vllm-rbln package. "
                        "Install with: pip install vllm-rbln"
                    )

        # Add extra options
        llm_kwargs.update(self.config.extra_options)
        llm_kwargs.update(kwargs)

        # Remove None values
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}

        self._llm = LLM(**llm_kwargs)
        logger.info(f"vLLM model loaded: {model_name_or_path}")
        return self._llm

    def start_server(self) -> None:
        """Start vLLM engine (library mode)"""
        if self._is_running:
            logger.warning("Server already running")
            return

        self._is_running = True
        logger.info("vLLM engine ready")

    def stop_server(self) -> None:
        """Stop vLLM server"""
        if self._server is not None:
            self._server.terminate()
            self._server.wait()
            self._server = None
        self._is_running = False
        logger.info("vLLM server stopped")

    def infer(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """
        Run inference with vLLM.

        Args:
            inputs: Text prompt or list of prompts
            **kwargs: Sampling parameters

        Returns:
            Generated text(s)
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
        )

        prompts = [inputs] if isinstance(inputs, str) else inputs
        outputs = self._llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append(completion.text)

        if isinstance(inputs, str):
            return results[0] if results else ""
        return results

    def infer_async(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """Async inference using HTTP API"""
        import requests

        if not self._is_running:
            raise RuntimeError("Server not running")

        url = f"http://{self.config.host}:{self.config.port}/v1/completions"

        payload = {
            "model": self.config.model_name,
            "prompt": inputs,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()

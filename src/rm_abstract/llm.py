"""
LLM Proxy Module - Unified interface for LLM inference across devices and engines.

This module provides a simple, hook-free approach to running LLM inference
on different hardware (GPU, NPU, CPU) with different serving engines (vLLM, Triton, TorchServe).

Usage:
    from rm_abstract import LLM, SamplingParams

    # Auto-detect best available device
    llm = LLM(model="gpt2")

    # Or specify device and engine explicitly
    llm = LLM(model="gpt2", device="nvidia", engine="vllm")
    llm = LLM(model="gpt2", device="cpu")  # Uses transformers fallback

    outputs = llm.generate(["Hello, I am"], SamplingParams(max_tokens=50))
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Get device from environment or default to auto."""
    return os.environ.get("RM_DEVICE", "auto")


def _get_engine() -> str:
    """Get serving engine from environment or default to vllm."""
    return os.environ.get("RM_ENGINE", "vllm")


@dataclass
class SamplingParams:
    """
    Sampling parameters for text generation.

    Compatible with vLLM's SamplingParams API.
    """

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    max_tokens: Optional[int] = 16
    stop: Optional[Union[str, List[str]]] = None
    skip_special_tokens: bool = True
    ignore_eos: bool = False

    def to_dict(self, include_best_of: bool = True) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_best_of: Whether to include best_of parameter.
                Set to False for vLLM v0.12+ which deprecated best_of.
        """
        result = {
            "n": self.n,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "skip_special_tokens": self.skip_special_tokens,
            "ignore_eos": self.ignore_eos,
        }
        if include_best_of and self.best_of is not None:
            result["best_of"] = self.best_of
        return result


@dataclass
class CompletionOutput:
    """Single completion output."""

    index: int
    text: str
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    finish_reason: Optional[str] = None


@dataclass
class RequestOutput:
    """Request output containing prompt and completions."""

    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool = True

    @property
    def output(self) -> Optional[CompletionOutput]:
        """Get first output (convenience property)."""
        return self.outputs[0] if self.outputs else None


class LLM:
    """
    Unified LLM interface for heterogeneous AI accelerators.

    Automatically routes to the appropriate backend based on device and engine selection.

    Supported devices:
        - auto: Auto-detect best available (NPU > GPU > CPU)
        - nvidia/gpu/cuda: NVIDIA GPU with vLLM
        - rbln: Rebellions ATOM NPU
        - furiosa: FuriosaAI RNGD NPU
        - cpu: CPU with transformers

    Supported engines:
        - vllm: High-performance LLM serving (default)
        - triton: NVIDIA Triton Inference Server
        - torchserve: PyTorch TorchServe

    Example:
        from rm_abstract import LLM, SamplingParams

        # Simple usage - auto-detect device
        llm = LLM(model="gpt2")
        outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
        print(outputs[0].outputs[0].text)

        # Explicit device selection
        llm = LLM(model="gpt2", device="nvidia", engine="vllm")
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        device: Optional[str] = None,
        engine: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Initialize LLM.

        Args:
            model: Model name or path (HuggingFace model ID)
            tokenizer: Tokenizer name (defaults to model)
            device: Device selection (auto, cpu, nvidia, rbln, furiosa)
            engine: Serving engine (vllm, triton, torchserve)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type (auto, float16, bfloat16, float32)
            trust_remote_code: Trust remote code in model
            **kwargs: Additional backend-specific arguments
        """
        self.model_name = model
        self.tokenizer_name = tokenizer or model
        self.device = device or _get_device()
        self.engine = engine or _get_engine()
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs

        self._backend: Optional[str] = None
        self._backend_llm: Any = None
        self._tokenizer: Any = None
        self._sampling_params_cls: Any = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate backend based on device and engine."""
        device = self._resolve_device(self.device)
        engine = self.engine.lower()
        logger.info(f"Initializing LLM with device: {device}, engine: {engine}")

        # Route based on engine first, then device
        if engine == "triton":
            self._init_triton_backend(device)
        elif engine == "torchserve":
            self._init_torchserve_backend(device)
        elif engine == "vllm":
            # vLLM engine - route based on device
            if device in ("nvidia", "gpu", "cuda"):
                self._init_vllm_gpu_backend()
            elif device == "rbln":
                self._init_vllm_rbln_backend()
            elif device == "furiosa":
                self._init_furiosa_backend()
            else:
                # CPU fallback for vllm engine
                self._init_cpu_backend()
        else:
            # Unknown engine, fallback to device-based selection
            if device in ("nvidia", "gpu", "cuda"):
                self._init_vllm_gpu_backend()
            elif device == "rbln":
                self._init_vllm_rbln_backend()
            elif device == "furiosa":
                self._init_furiosa_backend()
            else:
                self._init_cpu_backend()

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device.split(":")[0]  # Remove device index like nvidia:0

        # Auto-detection priority: NPU > GPU > CPU
        try:
            import rebel  # noqa: F401

            return "rbln"
        except ImportError:
            pass

        try:
            import furiosa_sdk  # noqa: F401

            return "furiosa"
        except ImportError:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                return "nvidia"
        except ImportError:
            pass

        return "cpu"

    def _init_vllm_gpu_backend(self):
        """Initialize vLLM backend for NVIDIA GPU."""
        self._backend = "vllm"
        try:
            import vllm

            self._backend_llm = vllm.LLM(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                **self.extra_kwargs,
            )
            self._sampling_params_cls = vllm.SamplingParams
            logger.info(f"Initialized vLLM GPU backend: {self.model_name}")

        except ImportError:
            logger.warning("vLLM not available, falling back to CPU")
            self._init_cpu_backend()

    def _init_vllm_rbln_backend(self):
        """Initialize vLLM backend for Rebellions NPU."""
        self._backend = "rbln"
        try:
            import vllm

            self._backend_llm = vllm.LLM(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                device="rbln",
                trust_remote_code=self.trust_remote_code,
                **self.extra_kwargs,
            )
            self._sampling_params_cls = vllm.SamplingParams
            logger.info(f"Initialized vLLM RBLN backend: {self.model_name}")

        except ImportError:
            # Fallback to optimum-rbln
            try:
                from optimum.rbln import RBLNLlamaForCausalLM
                from transformers import AutoTokenizer

                self._backend_llm = RBLNLlamaForCausalLM.from_pretrained(
                    self.model_name,
                    export=True,
                    trust_remote_code=self.trust_remote_code,
                )
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"Initialized optimum-rbln backend: {self.model_name}")

            except ImportError:
                logger.warning("RBLN backend not available, falling back to CPU")
                self._init_cpu_backend()

    def _init_furiosa_backend(self):
        """Initialize FuriosaAI NPU backend."""
        self._backend = "furiosa"
        try:
            from furiosa_llm import LLM as FuriosaLLM

            self._backend_llm = FuriosaLLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                **self.extra_kwargs,
            )
            logger.info(f"Initialized FuriosaAI backend: {self.model_name}")

        except ImportError:
            logger.warning("FuriosaAI backend not available, falling back to CPU")
            self._init_cpu_backend()

    def _init_triton_backend(self, device: str):
        """Initialize Triton Inference Server backend."""
        self._backend = "triton"
        try:
            from .engines.triton import TritonEngine
            from .engines.base import ServingConfig, ServingEngineType, DeviceTarget

            device_map = {
                "nvidia": DeviceTarget.GPU,
                "gpu": DeviceTarget.GPU,
                "cuda": DeviceTarget.GPU,
                "cpu": DeviceTarget.CPU,
            }
            target_device = device_map.get(device, DeviceTarget.CPU)

            config = ServingConfig(
                engine=ServingEngineType.TRITON,
                device=target_device,
                model_name=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
            )

            self._backend_llm = TritonEngine(config)
            self._backend_llm.load_model(self.model_name)
            logger.info(f"Initialized Triton backend: {self.model_name}")

        except ImportError as e:
            logger.warning(f"Triton backend not available ({e}), falling back to CPU")
            self._init_cpu_backend()

    def _init_torchserve_backend(self, device: str):
        """Initialize TorchServe backend."""
        self._backend = "torchserve"
        try:
            from .engines.torchserve import TorchServeEngine
            from .engines.base import ServingConfig, ServingEngineType, DeviceTarget

            device_map = {
                "nvidia": DeviceTarget.GPU,
                "gpu": DeviceTarget.GPU,
                "cuda": DeviceTarget.GPU,
                "cpu": DeviceTarget.CPU,
            }
            target_device = device_map.get(device, DeviceTarget.CPU)

            config = ServingConfig(
                engine=ServingEngineType.TORCHSERVE,
                device=target_device,
                model_name=self.model_name,
            )

            self._backend_llm = TorchServeEngine(config)
            self._backend_llm.load_model(self.model_name)
            logger.info(f"Initialized TorchServe backend: {self.model_name}")

        except ImportError as e:
            logger.warning(
                f"TorchServe backend not available ({e}), falling back to CPU"
            )
            self._init_cpu_backend()

    def _init_cpu_backend(self):
        """Initialize CPU backend using transformers."""
        self._backend = "cpu"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=self.trust_remote_code,
            )

            dtype_map = {
                "auto": torch.float32,
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.float32)

            self._backend_llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.trust_remote_code,
            )
            logger.info(f"Initialized CPU backend: {self.model_name}")

        except ImportError as e:
            raise RuntimeError(f"No backend available: {e}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> List[RequestOutput]:
        """
        Generate completions for the given prompts.

        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters
            **kwargs: Additional generation arguments

        Returns:
            List of RequestOutput objects
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if sampling_params is None:
            sampling_params = SamplingParams()

        if self._backend in ("triton", "torchserve"):
            return self._generate_engine_style(prompts, sampling_params, **kwargs)
        elif self._backend in ("vllm", "rbln") and hasattr(
            self._backend_llm, "generate"
        ):
            return self._generate_vllm_style(prompts, sampling_params, **kwargs)
        else:
            return self._generate_transformers_style(prompts, sampling_params, **kwargs)

    def _generate_vllm_style(
        self, prompts: List[str], sampling_params: SamplingParams, **kwargs
    ) -> List[RequestOutput]:
        """Generate using vLLM-style API."""
        if self._sampling_params_cls:
            # vLLM v0.12+ deprecated best_of parameter
            params_dict = sampling_params.to_dict(include_best_of=False)
            vllm_params = self._sampling_params_cls(**params_dict)
            outputs = self._backend_llm.generate(prompts, vllm_params, **kwargs)
        else:
            outputs = self._backend_llm.generate(
                prompts, sampling_params.to_dict(include_best_of=False), **kwargs
            )

        results = []
        for i, output in enumerate(outputs):
            if hasattr(output, "outputs"):
                completions = [
                    CompletionOutput(
                        index=j,
                        text=o.text,
                        token_ids=getattr(o, "token_ids", []),
                        finish_reason=getattr(o, "finish_reason", None),
                    )
                    for j, o in enumerate(output.outputs)
                ]
                results.append(
                    RequestOutput(
                        request_id=getattr(output, "request_id", str(i)),
                        prompt=prompts[i],
                        prompt_token_ids=getattr(output, "prompt_token_ids", []),
                        outputs=completions,
                    )
                )
            else:
                results.append(
                    RequestOutput(
                        request_id=str(i),
                        prompt=prompts[i],
                        prompt_token_ids=[],
                        outputs=[CompletionOutput(index=0, text=str(output))],
                    )
                )

        return results

    def _generate_transformers_style(
        self, prompts: List[str], sampling_params: SamplingParams, **kwargs
    ) -> List[RequestOutput]:
        """Generate using transformers-style API."""
        import torch

        results = []
        for i, prompt in enumerate(prompts):
            inputs = self._tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                output_ids = self._backend_llm.generate(
                    inputs.input_ids,
                    max_new_tokens=sampling_params.max_tokens or 16,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    top_k=sampling_params.top_k if sampling_params.top_k > 0 else None,
                    do_sample=sampling_params.temperature > 0,
                    num_return_sequences=sampling_params.n,
                    **kwargs,
                )

            completions = []
            for j, ids in enumerate(output_ids):
                new_ids = ids[inputs.input_ids.shape[1] :]
                text = self._tokenizer.decode(
                    new_ids,
                    skip_special_tokens=sampling_params.skip_special_tokens,
                )
                completions.append(
                    CompletionOutput(
                        index=j,
                        text=text,
                        token_ids=new_ids.tolist(),
                        finish_reason="stop",
                    )
                )

            results.append(
                RequestOutput(
                    request_id=str(i),
                    prompt=prompt,
                    prompt_token_ids=inputs.input_ids[0].tolist(),
                    outputs=completions,
                )
            )

        return results

    def _generate_engine_style(
        self, prompts: List[str], sampling_params: SamplingParams, **kwargs
    ) -> List[RequestOutput]:
        """Generate using ServingEngine API (Triton/TorchServe)."""
        if self._backend_llm is None:
            raise RuntimeError("Engine not initialized")

        results = []
        for i, prompt in enumerate(prompts):
            output = self._backend_llm.infer(
                prompt,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                max_tokens=sampling_params.max_tokens,
                **kwargs,
            )

            if isinstance(output, str):
                text = output
            elif isinstance(output, dict):
                text = output.get("text", output.get("generated_text", str(output)))
            else:
                text = str(output)

            results.append(
                RequestOutput(
                    request_id=str(i),
                    prompt=prompt,
                    prompt_token_ids=[],
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=text,
                            token_ids=[],
                            finish_reason="stop",
                        )
                    ],
                )
            )

        return results

    @property
    def backend(self) -> str:
        """Get current backend name."""
        return self._backend or "unknown"

    @property
    def engine_type(self) -> str:
        """Get current engine type."""
        return self.engine

    def __repr__(self) -> str:
        return f"LLM(model={self.model_name!r}, device={self.device!r}, engine={self.engine!r}, backend={self._backend!r})"

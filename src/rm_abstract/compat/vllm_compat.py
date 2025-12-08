"""
vLLM Compatibility Layer

Provides drop-in replacement for vLLM that routes to appropriate backend
based on RM_DEVICE environment variable or explicit configuration.

Usage:
    # Original code (no modification needed with auto-hook)
    from vllm import LLM, SamplingParams

    # Or explicit import
    from rm_abstract.compat import LLM, SamplingParams

    llm = LLM(model="gpt2")
    output = llm.generate("Hello", SamplingParams())
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Get device from environment or auto-detect."""
    return os.environ.get("RM_DEVICE", "auto")


def _get_engine() -> str:
    """Get serving engine from environment."""
    return os.environ.get("RM_ENGINE", "vllm")


@dataclass
class SamplingParams:
    """
    vLLM-compatible sampling parameters.

    Works across all backends with automatic parameter translation.
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backend consumption."""
        return {
            "n": self.n,
            "best_of": self.best_of or self.n,
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
    """vLLM-compatible request output."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    outputs: List[CompletionOutput]
    finished: bool = True

    @property
    def output(self) -> CompletionOutput:
        """Get first output (convenience property)."""
        return self.outputs[0] if self.outputs else None


class LLM:
    """
    vLLM-compatible LLM interface.

    Automatically routes to the appropriate backend based on:
    - RM_DEVICE environment variable
    - Explicit device parameter
    - Auto-detection (NPU > GPU > CPU priority)

    Example:
        # Use environment variable
        # RM_DEVICE=rbln python script.py

        llm = LLM(model="meta-llama/Llama-2-7b-hf")
        outputs = llm.generate(["Hello, world!"], SamplingParams())

        # Or explicit device selection
        llm = LLM(model="gpt2", device="cpu")
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        device: Optional[str] = None,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        **kwargs
    ):
        """
        Initialize LLM.

        Args:
            model: Model name or path (HuggingFace model ID)
            tokenizer: Tokenizer name (defaults to model)
            device: Device selection (auto, cpu, nvidia, rbln, furiosa)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            dtype: Data type (auto, float16, bfloat16, float32)
            trust_remote_code: Trust remote code in model
            **kwargs: Additional backend-specific arguments
        """
        self.model_name = model
        self.tokenizer_name = tokenizer or model
        self.device = device or _get_device()
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.extra_kwargs = kwargs

        self._backend = None
        self._backend_llm = None
        self._tokenizer = None

        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the appropriate backend based on device selection."""
        device = self._resolve_device(self.device)
        logger.info(f"Initializing LLM with device: {device}")

        if device in ("nvidia", "gpu", "cuda"):
            self._init_vllm_backend()
        elif device == "rbln":
            self._init_rbln_backend()
        elif device == "furiosa":
            self._init_furiosa_backend()
        else:
            self._init_cpu_backend()

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device.split(":")[0]  # Remove device index

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

    def _init_vllm_backend(self):
        """Initialize vLLM backend for NVIDIA GPU."""
        self._backend = "vllm"
        try:
            from vllm import LLM as VLLMEngine
            from vllm import SamplingParams as VLLMSamplingParams

            self._backend_llm = VLLMEngine(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                tensor_parallel_size=self.tensor_parallel_size,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
                **self.extra_kwargs
            )
            self._sampling_params_cls = VLLMSamplingParams
            logger.info(f"Initialized vLLM backend for model: {self.model_name}")

        except ImportError:
            logger.warning("vLLM not available, falling back to CPU")
            self._init_cpu_backend()

    def _init_rbln_backend(self):
        """Initialize Rebellions NPU backend."""
        self._backend = "rbln"
        try:
            # Try vllm-rbln first
            from vllm import LLM as VLLMRBLNEngine

            self._backend_llm = VLLMRBLNEngine(
                model=self.model_name,
                tokenizer=self.tokenizer_name,
                device="rbln",
                trust_remote_code=self.trust_remote_code,
                **self.extra_kwargs
            )
            logger.info(f"Initialized vLLM-RBLN backend for model: {self.model_name}")

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
                logger.info(f"Initialized optimum-rbln backend for model: {self.model_name}")

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
                **self.extra_kwargs
            )
            logger.info(f"Initialized FuriosaAI backend for model: {self.model_name}")

        except ImportError:
            logger.warning("FuriosaAI backend not available, falling back to CPU")
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
            logger.info(f"Initialized CPU backend for model: {self.model_name}")

        except ImportError as e:
            raise RuntimeError(f"No backend available: {e}")

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
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

        if self._backend in ("vllm", "rbln") and hasattr(self._backend_llm, "generate"):
            return self._generate_vllm_style(prompts, sampling_params, **kwargs)
        else:
            return self._generate_transformers_style(prompts, sampling_params, **kwargs)

    def _generate_vllm_style(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        **kwargs
    ) -> List[RequestOutput]:
        """Generate using vLLM-style API."""
        # Convert to vLLM SamplingParams if available
        if hasattr(self, "_sampling_params_cls"):
            vllm_params = self._sampling_params_cls(**sampling_params.to_dict())
            outputs = self._backend_llm.generate(prompts, vllm_params, **kwargs)
        else:
            outputs = self._backend_llm.generate(prompts, sampling_params.to_dict(), **kwargs)

        # Convert to our RequestOutput format
        results = []
        for i, output in enumerate(outputs):
            if hasattr(output, "outputs"):
                # Native vLLM output
                completions = [
                    CompletionOutput(
                        index=j,
                        text=o.text,
                        token_ids=getattr(o, "token_ids", []),
                        finish_reason=getattr(o, "finish_reason", None),
                    )
                    for j, o in enumerate(output.outputs)
                ]
                results.append(RequestOutput(
                    request_id=getattr(output, "request_id", str(i)),
                    prompt=prompts[i],
                    prompt_token_ids=getattr(output, "prompt_token_ids", []),
                    outputs=completions,
                ))
            else:
                # Simple string output
                results.append(RequestOutput(
                    request_id=str(i),
                    prompt=prompts[i],
                    prompt_token_ids=[],
                    outputs=[CompletionOutput(index=0, text=str(output))],
                ))

        return results

    def _generate_transformers_style(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        **kwargs
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
                    **kwargs
                )

            # Decode outputs
            completions = []
            for j, ids in enumerate(output_ids):
                # Remove prompt tokens
                new_ids = ids[inputs.input_ids.shape[1]:]
                text = self._tokenizer.decode(
                    new_ids,
                    skip_special_tokens=sampling_params.skip_special_tokens,
                )
                completions.append(CompletionOutput(
                    index=j,
                    text=text,
                    token_ids=new_ids.tolist(),
                    finish_reason="stop",
                ))

            results.append(RequestOutput(
                request_id=str(i),
                prompt=prompt,
                prompt_token_ids=inputs.input_ids[0].tolist(),
                outputs=completions,
            ))

        return results

    @property
    def backend(self) -> str:
        """Get current backend name."""
        return self._backend

    def __repr__(self) -> str:
        return f"LLM(model={self.model_name!r}, backend={self._backend!r})"

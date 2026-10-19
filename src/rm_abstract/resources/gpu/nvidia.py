"""
NVIDIA GPU Resource

GPU resource implementation using vLLM for high-performance inference.
"""

import logging
from typing import Any, Dict, Optional

from ...common.input_adapter import InputAdapter
from ...common.output_adapter import OutputAdapter
from ..base import Resource, ResourceCapability, ResourceInfo, ResourceType

logger = logging.getLogger(__name__)


class NvidiaGPU(Resource):
    """
    NVIDIA GPU Resource

    Uses vLLM for high-performance LLM inference on NVIDIA GPUs.

    Example:
        gpu = NvidiaGPU(device_id=0)
        gpu.initialize()

        model = gpu.prepare_model("meta-llama/Llama-2-7b-hf")
        output = gpu.execute(model, "Hello, world!")

        gpu.cleanup()
    """

    RESOURCE_TYPE = ResourceType.GPU
    VENDOR = "nvidia"
    NAME = "NVIDIA GPU"

    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)
        self._llm_engine = None
        self._model_name: Optional[str] = None
        self._input_adapter = InputAdapter()
        self._output_adapter = OutputAdapter()

    @staticmethod
    def is_available() -> bool:
        """Check CUDA and vLLM availability"""
        try:
            import torch

            if not torch.cuda.is_available():
                return False

            # Check vLLM import
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self) -> None:
        """Initialize GPU resource"""
        if self._initialized:
            return

        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            torch.cuda.set_device(self.device_id)
            self._initialized = True
            logger.info(f"NVIDIA GPU initialized on cuda:{self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA GPU: {e}")
            raise

    def prepare_model(self, model: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model using vLLM.

        Args:
            model: Model name/path or PyTorch model
            config: vLLM configuration options

        Returns:
            vLLM LLM instance
        """
        config = config or {}

        # Extract model name
        model_name = self._get_model_name(model)

        try:
            from vllm import LLM

            self._llm_engine = LLM(
                model=model_name,
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
                dtype=config.get("dtype", "auto"),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
                trust_remote_code=config.get("trust_remote_code", True),
            )
            self._model_name = model_name

            # Set up tokenizer for adapters
            tokenizer = self._get_tokenizer(model_name)
            if tokenizer:
                self._input_adapter.set_tokenizer(tokenizer)
                self._output_adapter.set_tokenizer(tokenizer)

            logger.info(f"vLLM engine created for model: {model_name}")
            return self._llm_engine

        except Exception as e:
            logger.error(f"Failed to create vLLM engine: {e}")
            raise

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference with vLLM.

        Args:
            model: vLLM LLM instance
            inputs: Input data (string, token IDs, etc.)
            **kwargs: Generation options

        Returns:
            Generated output
        """
        from vllm import SamplingParams

        # Extract proxy metadata
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        # Get tokenizer from original model if available
        if original_model:
            tokenizer = self._input_adapter.get_tokenizer_from_model(original_model)
            if tokenizer:
                self._input_adapter.set_tokenizer(tokenizer)
                self._output_adapter.set_tokenizer(tokenizer)

        # Adapt inputs to prompts
        adapted = self._input_adapter.adapt(inputs, **kwargs)
        prompts = adapted.prompts or [str(inputs)]

        # Handle sampling parameters
        do_sample = kwargs.pop("do_sample", True)
        temperature = kwargs.get("temperature", 0.7 if do_sample else 0.0)
        if not do_sample:
            temperature = 0.0

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=kwargs.get("top_p", 1.0 if not do_sample else 0.95),
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
        )

        # Execute inference
        outputs = model.generate(prompts, sampling_params)

        # Convert output format if needed
        if proxy_method == "generate":
            return self._output_adapter.to_hf_format(outputs, inputs)

        return outputs

    def get_info(self) -> ResourceInfo:
        """Return GPU resource information"""
        try:
            import torch

            props = torch.cuda.get_device_properties(self.device_id)
            return ResourceInfo(
                resource_type=ResourceType.GPU,
                device_id=self.device_id,
                name=props.name,
                vendor=self.VENDOR,
                memory_total=props.total_memory,
                memory_available=torch.cuda.memory_reserved(self.device_id),
                extra={
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                },
            )
        except Exception:
            return ResourceInfo(
                resource_type=ResourceType.GPU,
                device_id=self.device_id,
                name="Unknown GPU",
                vendor=self.VENDOR,
            )

    def get_capability(self) -> ResourceCapability:
        """Return GPU capabilities"""
        return ResourceCapability(
            supports_fp16=True,
            supports_fp32=True,
            supports_int8=True,
            supports_continuous_batching=True,
            supports_tensor_parallel=True,
        )

    def cleanup(self) -> None:
        """Cleanup GPU resources"""
        self._llm_engine = None
        self._model_name = None

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        super().cleanup()

    def _get_model_name(self, model: Any) -> str:
        """Extract model name from various input types"""
        if isinstance(model, str):
            return model
        elif hasattr(model, "name_or_path"):
            return model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            return model.config.name_or_path
        else:
            raise ValueError("Cannot determine model name/path")

    def _get_tokenizer(self, model_name: str) -> Optional[Any]:
        """Get tokenizer for a model"""
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(model_name)
        except Exception:
            return None

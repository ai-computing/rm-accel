"""
Rebellions ATOM NPU Resource

NPU resource implementation for Rebellions ATOM chips.
Supports two modes: vLLM-RBLN and Optimum-RBLN.
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from ...common.input_adapter import InputAdapter
from ...common.model_converter import ModelConverter
from ...common.output_adapter import OutputAdapter
from ..base import NPUResource, ResourceCapability, ResourceInfo, ResourceType

logger = logging.getLogger(__name__)


class RBLNMode(str, Enum):
    """RBLN inference mode"""

    VLLM = "vllm"  # vLLM-RBLN for high-performance serving
    OPTIMUM = "optimum"  # Optimum-RBLN for HuggingFace integration
    AUTO = "auto"  # Auto-detect based on available packages


class RebellionsATOM(NPUResource):
    """
    Rebellions ATOM NPU Resource

    Supports two inference modes:
    1. vLLM-RBLN: High-performance LLM serving via vLLM integration
    2. Optimum-RBLN: HuggingFace model integration via optimum-rbln

    Example:
        # Auto mode (recommended)
        npu = RebellionsATOM(device_id=0, mode="auto")
        npu.initialize()

        model = npu.prepare_model("meta-llama/Llama-2-7b-hf")
        output = npu.execute(model, "Hello!")

        npu.cleanup()
    """

    RESOURCE_TYPE = ResourceType.NPU
    VENDOR = "rebellions"
    NAME = "Rebellions ATOM"

    def __init__(
        self,
        device_id: int = 0,
        mode: Literal["vllm", "optimum", "auto"] = "auto",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device_id, cache_dir, **kwargs)
        self._requested_mode = RBLNMode(mode)
        self._active_mode: Optional[RBLNMode] = None
        self._llm_engine = None
        self._compiled_model = None
        self._model_name: Optional[str] = None
        self._tokenizer = None
        self._input_adapter = InputAdapter()
        self._output_adapter = OutputAdapter()
        self._model_converter = ModelConverter(cache_dir)

        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = os.path.join(Path.home(), ".rm_abstract", "cache", "rbln")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @property
    def compiled_model_extension(self) -> str:
        return "rbln"

    @property
    def mode(self) -> Optional[RBLNMode]:
        """Current active mode"""
        return self._active_mode

    @staticmethod
    def is_available() -> bool:
        """Check RBLN SDK and NPU availability"""
        vllm_available = RebellionsATOM._check_vllm_rbln_static()
        optimum_available = RebellionsATOM._check_optimum_rbln_static()

        if not (vllm_available or optimum_available):
            logger.debug("No RBLN SDK found")
            return False

        return RebellionsATOM._check_npu_device_static()

    @staticmethod
    def _check_vllm_rbln_static() -> bool:
        """Check if vLLM-RBLN is available (static version)"""
        try:
            import vllm_rbln  # noqa: F401

            return True
        except ImportError:
            pass

        # Check rebel SDK + vLLM
        try:
            import rebel  # noqa: F401
            import vllm

            return True
        except ImportError:
            pass

        # Check rbln SDK + vLLM
        try:
            import rbln  # noqa: F401
            import vllm  # noqa: F401

            return True
        except ImportError:
            pass

        return False

    @staticmethod
    def _check_optimum_rbln_static() -> bool:
        """Check if Optimum-RBLN is available (static version)"""
        try:
            from optimum.rbln import RBLNModelForCausalLM  # noqa: F401

            return True
        except ImportError:
            try:
                import optimum_rbln  # noqa: F401

                return True
            except ImportError:
                return False

    @staticmethod
    def _check_npu_device_static() -> bool:
        """Check if RBLN NPU device is available (static version)"""
        try:
            import rebel

            devices = rebel.get_devices()
            return len(devices) > 0
        except ImportError:
            pass

        try:
            import rbln

            devices = rbln.get_devices()
            return len(devices) > 0
        except ImportError:
            pass

        return False

    def _check_vllm_rbln(self) -> bool:
        """Check if vLLM-RBLN is available"""
        return RebellionsATOM._check_vllm_rbln_static()

    def _check_optimum_rbln(self) -> bool:
        """Check if Optimum-RBLN is available"""
        return RebellionsATOM._check_optimum_rbln_static()

    def _check_npu_device(self) -> bool:
        """Check if RBLN NPU device is available"""
        try:
            import rebel

            devices = rebel.get_devices()
            return len(devices) > self.device_id
        except ImportError:
            pass

        try:
            import rbln

            devices = rbln.get_devices()
            return len(devices) > self.device_id
        except ImportError:
            pass

        return False

    def _determine_mode(self) -> RBLNMode:
        """Determine which mode to use"""
        if self._requested_mode == RBLNMode.AUTO:
            if self._check_vllm_rbln():
                return RBLNMode.VLLM
            elif self._check_optimum_rbln():
                return RBLNMode.OPTIMUM
            else:
                raise RuntimeError("No RBLN SDK available")
        else:
            if self._requested_mode == RBLNMode.VLLM and not self._check_vllm_rbln():
                raise RuntimeError("vLLM-RBLN not available")
            if (
                self._requested_mode == RBLNMode.OPTIMUM
                and not self._check_optimum_rbln()
            ):
                raise RuntimeError("Optimum-RBLN not available")
            return self._requested_mode

    def initialize(self) -> None:
        """Initialize RBLN resource"""
        if self._initialized:
            return

        try:
            self._active_mode = self._determine_mode()
            self._initialized = True
            logger.info(
                f"Rebellions ATOM initialized in {self._active_mode.value} mode "
                f"on device {self.device_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Rebellions ATOM: {e}")
            raise

    def prepare_model(self, model: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Prepare model for RBLN NPU"""
        if not self._initialized:
            self.initialize()

        config = config or {}

        # Allow mode override per model
        if "rbln_mode" in config:
            requested_mode = RBLNMode(config.pop("rbln_mode"))
            if requested_mode != self._active_mode:
                logger.info(f"Switching mode to {requested_mode.value}")
                self._active_mode = requested_mode

        if self._active_mode == RBLNMode.VLLM:
            return self._prepare_vllm(model, config)
        else:
            return self._prepare_optimum(model, config)

    def _prepare_vllm(self, model: Any, config: Dict[str, Any]) -> Any:
        """Prepare model using vLLM-RBLN"""
        model_name = self._get_model_name(model)

        try:
            from vllm import LLM

            llm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": config.get("tensor_parallel_size", 1),
                "dtype": config.get("dtype", "auto"),
                "trust_remote_code": config.get("trust_remote_code", True),
                "max_num_seqs": config.get("max_num_seqs", 16),
            }

            if config.get("max_model_len"):
                llm_kwargs["max_model_len"] = config["max_model_len"]

            # Add RBLN device
            try:
                import vllm_rbln  # noqa: F401

                llm_kwargs["device"] = "rbln"
            except ImportError:
                try:

                    llm_kwargs["device"] = "rbln"
                except Exception:
                    pass

            self._llm_engine = LLM(**llm_kwargs)
            self._model_name = model_name

            # Set up adapters
            self._setup_tokenizer(model_name)

            logger.info(f"vLLM-RBLN engine created for: {model_name}")
            return self._llm_engine

        except Exception as e:
            logger.error(f"Failed to create vLLM-RBLN engine: {e}")
            raise

    def _prepare_optimum(self, model: Any, config: Dict[str, Any]) -> Any:
        """Prepare model using Optimum-RBLN"""
        model_name = self._get_model_name(model)

        try:
            from optimum.rbln import RBLNModelForCausalLM
            from transformers import AutoTokenizer

            export = config.get("export", True)

            if export:
                self._compiled_model = RBLNModelForCausalLM.from_pretrained(
                    model_name,
                    export=True,
                    rbln_device_id=self.device_id,
                    rbln_batch_size=config.get("batch_size", 1),
                    rbln_max_seq_len=config.get("max_seq_len", 2048),
                )
            else:
                self._compiled_model = RBLNModelForCausalLM.from_pretrained(
                    model_name,
                    rbln_device_id=self.device_id,
                )

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model_name = model_name

            # Set up adapters
            self._input_adapter.set_tokenizer(self._tokenizer)
            self._output_adapter.set_tokenizer(self._tokenizer)

            logger.info(f"Optimum-RBLN model loaded: {model_name}")
            return self._compiled_model

        except Exception as e:
            logger.error(f"Failed to load Optimum-RBLN model: {e}")
            raise

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute inference on RBLN NPU"""
        if self._active_mode == RBLNMode.VLLM:
            return self._execute_vllm(model, inputs, **kwargs)
        else:
            return self._execute_optimum(model, inputs, **kwargs)

    def _execute_vllm(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute with vLLM-RBLN"""
        from vllm import SamplingParams

        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)

        # Update tokenizer if needed
        if original_model:
            tokenizer = self._input_adapter.get_tokenizer_from_model(original_model)
            if tokenizer:
                self._input_adapter.set_tokenizer(tokenizer)
                self._output_adapter.set_tokenizer(tokenizer)

        # Adapt inputs
        adapted = self._input_adapter.adapt(inputs, **kwargs)
        prompts = adapted.prompts or [str(inputs)]

        # Sampling parameters
        do_sample = kwargs.pop("do_sample", True)
        temperature = kwargs.get("temperature", 0.7 if do_sample else 0.0)
        if not do_sample:
            temperature = 0.0

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=kwargs.get("top_p", 1.0 if not do_sample else 0.95),
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
        )

        outputs = model.generate(prompts, sampling_params)

        if proxy_method == "generate":
            return self._output_adapter.to_hf_format(outputs, inputs)

        return outputs

    def _execute_optimum(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute with Optimum-RBLN"""
        import torch

        proxy_method = kwargs.pop("_proxy_method", None)
        kwargs.pop("original_model", None)

        # Adapt inputs
        if isinstance(inputs, str):
            inputs = self._tokenizer(inputs, return_tensors="pt")
        elif not isinstance(inputs, dict) and not hasattr(inputs, "input_ids"):
            adapted = self._input_adapter.adapt(inputs)
            if adapted.token_ids is not None:
                inputs = {"input_ids": adapted.token_ids}

        # Execute
        if proxy_method == "generate":
            max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 128))
            do_sample = kwargs.get("do_sample", False)
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)

            input_dict = (
                inputs if isinstance(inputs, dict) else {"input_ids": inputs.input_ids}
            )
            outputs = model.generate(
                **input_dict,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            return outputs
        else:
            with torch.no_grad():
                input_dict = (
                    inputs
                    if isinstance(inputs, dict)
                    else {"input_ids": inputs.input_ids}
                )
                return model(**input_dict)

    def compile_model(self, model: Any, **kwargs) -> Any:
        """Compile model for RBLN (used with ONNX path)"""
        # This is handled internally by optimum-rbln
        raise NotImplementedError(
            "Use prepare_model() instead. "
            "RBLN compilation is handled by optimum-rbln or vLLM-RBLN."
        )

    def load_compiled_model(self, path: str) -> Any:
        """Load compiled RBLN model"""
        try:
            from optimum.rbln import RBLNModelForCausalLM

            return RBLNModelForCausalLM.from_pretrained(
                path, rbln_device_id=self.device_id
            )
        except Exception as e:
            logger.error(f"Failed to load compiled model: {e}")
            raise

    def save_compiled_model(self, model: Any, path: str) -> None:
        """Save compiled RBLN model"""
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
            logger.info(f"Model saved to: {path}")
        else:
            raise ValueError("Model does not support save_pretrained()")

    def get_info(self) -> ResourceInfo:
        """Return RBLN NPU information"""
        extra_info = {
            "mode": self._active_mode.value if self._active_mode else "not_initialized",
        }

        sdk_version = self._get_sdk_version()
        if sdk_version:
            extra_info["sdk_version"] = sdk_version

        return ResourceInfo(
            resource_type=ResourceType.NPU,
            device_id=self.device_id,
            name=self.NAME,
            vendor=self.VENDOR,
            extra=extra_info,
        )

    def get_capability(self) -> ResourceCapability:
        """Return RBLN capabilities"""
        return ResourceCapability(
            supports_fp16=True,
            supports_int8=True,
            supports_continuous_batching=(self._active_mode == RBLNMode.VLLM),
            supports_tensor_parallel=True,
        )

    def cleanup(self) -> None:
        """Cleanup RBLN resources"""
        self._llm_engine = None
        self._compiled_model = None
        self._tokenizer = None
        self._model_name = None
        self._active_mode = None
        super().cleanup()

    def _get_model_name(self, model: Any) -> str:
        """Extract model name"""
        if isinstance(model, str):
            return model
        elif hasattr(model, "name_or_path"):
            return model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            return model.config.name_or_path
        else:
            raise ValueError("Cannot determine model name/path")

    def _setup_tokenizer(self, model_name: str) -> None:
        """Set up tokenizer for adapters"""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._input_adapter.set_tokenizer(self._tokenizer)
            self._output_adapter.set_tokenizer(self._tokenizer)
        except Exception:
            pass

    def _get_sdk_version(self) -> Optional[str]:
        """Get RBLN SDK version"""
        try:
            import rebel

            return getattr(rebel, "__version__", "unknown")
        except ImportError:
            pass

        try:
            import rbln

            return getattr(rbln, "__version__", "unknown")
        except ImportError:
            pass

        return None

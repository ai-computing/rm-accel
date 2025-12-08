"""
Rebellions ATOM NPU Backend

Supports two inference modes:
1. vLLM-RBLN: High-performance LLM serving via vLLM integration
2. Optimum-RBLN: HuggingFace model integration via optimum-rbln

Reference: https://docs.rbln.ai/latest/
"""

from typing import Any, Dict, Optional, Literal
from enum import Enum
import logging

from ..base import NPUBackendBase
from ....core.backend import Backend, DeviceType, DeviceInfo

logger = logging.getLogger(__name__)


class RBLNMode(str, Enum):
    """RBLN Backend inference mode"""
    VLLM = "vllm"       # vLLM-RBLN for high-performance LLM serving
    OPTIMUM = "optimum"  # Optimum-RBLN for HuggingFace model integration
    AUTO = "auto"        # Auto-detect based on available packages


class RBLNBackend(Backend):
    """
    Rebellions ATOM NPU Backend

    Supports both vLLM-RBLN and Optimum-RBLN modes for flexible LLM inference.
    
    Args:
        device_id: NPU device ID (default: 0)
        mode: Inference mode - "vllm", "optimum", or "auto" (default: "auto")
        **kwargs: Additional backend-specific options
        
    Example:
        # Auto mode (recommended)
        backend = RBLNBackend(device_id=0, mode="auto")
        
        # vLLM mode for serving
        backend = RBLNBackend(device_id=0, mode="vllm")
        
        # Optimum mode for HuggingFace integration
        backend = RBLNBackend(device_id=0, mode="optimum")
    """

    def __init__(
        self,
        device_id: int = 0,
        mode: Literal["vllm", "optimum", "auto"] = "auto",
        **kwargs,
    ):
        super().__init__(device_id, **kwargs)
        self._requested_mode = RBLNMode(mode)
        self._active_mode: Optional[RBLNMode] = None
        self._backend_impl: Optional[Backend] = None
        self._model_name: Optional[str] = None
        
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.NPU

    @property
    def name(self) -> str:
        return "rbln"

    @property
    def mode(self) -> Optional[RBLNMode]:
        """Current active mode"""
        return self._active_mode

    def is_available(self) -> bool:
        """Check RBLN SDK and NPU availability"""
        # Check for any RBLN package
        vllm_available = self._check_vllm_rbln()
        optimum_available = self._check_optimum_rbln()
        
        if not (vllm_available or optimum_available):
            logger.debug("No RBLN SDK found (neither vllm-rbln nor optimum-rbln)")
            return False
        
        # Check NPU device
        return self._check_npu_device()
    
    def _check_vllm_rbln(self) -> bool:
        """Check if vLLM-RBLN is available (not just regular vLLM)"""
        try:
            # Check for vllm-rbln specific package
            import vllm_rbln
            return True
        except ImportError:
            pass
        
        # Check if RBLN SDK is installed (required for vLLM-RBLN)
        try:
            import rebel
            # If rebel SDK exists, check if vLLM is also available
            try:
                import vllm
                return True
            except ImportError:
                return False
        except ImportError:
            pass
        
        try:
            import rbln
            try:
                import vllm
                return True
            except ImportError:
                return False
        except ImportError:
            pass
        
        return False
    
    def _check_optimum_rbln(self) -> bool:
        """Check if Optimum-RBLN is available"""
        try:
            from optimum.rbln import RBLNModelForCausalLM
            return True
        except ImportError:
            try:
                # Alternative import path
                import optimum_rbln
                return True
            except ImportError:
                return False
    
    def _check_npu_device(self) -> bool:
        """Check if RBLN NPU device is available"""
        try:
            import rebel
            devices = rebel.get_devices()
            return len(devices) > self.device_id
        except ImportError:
            # Try alternative package names
            try:
                import rbln
                devices = rbln.get_devices()
                return len(devices) > self.device_id
            except ImportError:
                pass
        except Exception as e:
            logger.debug(f"NPU device check failed: {e}")
        
        # If we can't check devices, NPU is not available
        logger.debug("Cannot detect RBLN NPU devices - SDK not found or no devices")
            return False

    def _determine_mode(self) -> RBLNMode:
        """Determine which mode to use based on availability and preference"""
        if self._requested_mode == RBLNMode.AUTO:
            # Prefer vLLM for LLM serving if available
            if self._check_vllm_rbln():
                return RBLNMode.VLLM
            elif self._check_optimum_rbln():
                return RBLNMode.OPTIMUM
            else:
                raise RuntimeError("No RBLN SDK available")
        else:
            # Use requested mode
            if self._requested_mode == RBLNMode.VLLM and not self._check_vllm_rbln():
                raise RuntimeError("vLLM-RBLN not available")
            if self._requested_mode == RBLNMode.OPTIMUM and not self._check_optimum_rbln():
                raise RuntimeError("Optimum-RBLN not available")
            return self._requested_mode
    
    def initialize(self) -> None:
        """Initialize RBLN backend"""
        if self._initialized:
            return

        try:
            self._active_mode = self._determine_mode()
            
            if self._active_mode == RBLNMode.VLLM:
                self._backend_impl = RBLNVLLMBackend(self.device_id)
            else:
                self._backend_impl = RBLNOptimumBackend(self.device_id)
            
            self._backend_impl.initialize()
            self._initialized = True
            
            logger.info(f"RBLN backend initialized in {self._active_mode.value} mode on device {self.device_id}")

        except Exception as e:
            logger.error(f"Failed to initialize RBLN backend: {e}")
            raise

    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """Prepare model for RBLN NPU"""
        if not self._initialized:
            self.initialize()
        
        config = model_config or {}
        # Allow overriding mode per model
        if "rbln_mode" in config:
            requested_mode = RBLNMode(config.pop("rbln_mode"))
            if requested_mode != self._active_mode:
                logger.info(f"Switching RBLN mode from {self._active_mode.value} to {requested_mode.value}")
                self._active_mode = requested_mode
                if self._active_mode == RBLNMode.VLLM:
                    self._backend_impl = RBLNVLLMBackend(self.device_id)
                else:
                    self._backend_impl = RBLNOptimumBackend(self.device_id)
                self._backend_impl.initialize()
        
        return self._backend_impl.prepare_model(model, config)
    
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute inference on RBLN NPU"""
        if self._backend_impl is None:
            raise RuntimeError("Backend not initialized")
        return self._backend_impl.execute(model, inputs, **kwargs)
    
    def get_device_info(self) -> DeviceInfo:
        """Return RBLN NPU device information"""
        extra_info = {
            "mode": self._active_mode.value if self._active_mode else "not_initialized",
        }
        
        try:
            import rebel
            devices = rebel.get_devices()
            if self.device_id < len(devices):
                extra_info["sdk_version"] = getattr(rebel, "__version__", "unknown")
        except ImportError:
            try:
                import rbln
                extra_info["sdk_version"] = getattr(rbln, "__version__", "unknown")
            except ImportError:
                pass
        
        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="Rebellions ATOM",
            vendor="Rebellions",
            extra=extra_info,
        )
    
    def cleanup(self) -> None:
        """Cleanup RBLN resources"""
        if self._backend_impl is not None:
            self._backend_impl.cleanup()
            self._backend_impl = None
        self._active_mode = None
        super().cleanup()


class RBLNVLLMBackend(Backend):
    """
    vLLM-RBLN Backend Implementation
    
    High-performance LLM serving using vLLM with RBLN NPU acceleration.
    Reference: https://docs.rbln.ai/latest/model_serving/vllm/
    """
    
    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)
        self._llm_engine = None
        self._model_name: Optional[str] = None
    
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.NPU
    
    @property
    def name(self) -> str:
        return "rbln_vllm"
    
    def is_available(self) -> bool:
        try:
            import vllm
            return True
        except ImportError:
            return False
    
    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info(f"RBLN vLLM backend initialized on device {self.device_id}")
    
    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model using vLLM-RBLN

        Args:
            model: Model name/path or HuggingFace model
            model_config: vLLM configuration options
        """
        config = model_config or {}
        
        # Extract model name
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            model_name = model.config.name_or_path
        else:
            raise ValueError("Cannot determine model name/path")
        
        try:
            from vllm import LLM
            
            # Build vLLM-RBLN configuration
            llm_kwargs = {
                "model": model_name,
                "tensor_parallel_size": config.get("tensor_parallel_size", 1),
                "dtype": config.get("dtype", "auto"),
                "trust_remote_code": config.get("trust_remote_code", True),
                "max_num_seqs": config.get("max_num_seqs", 16),
            }
            
            # Add max_model_len if specified
            if config.get("max_model_len"):
                llm_kwargs["max_model_len"] = config["max_model_len"]
            
            # Add RBLN device only if vLLM-RBLN is actually available
            try:
                import vllm_rbln
                llm_kwargs["device"] = "rbln"
                logger.debug("Using vLLM-RBLN with device='rbln'")
            except ImportError:
                # Check if vLLM supports RBLN device natively
                try:
                    from vllm.config import DeviceConfig
                    llm_kwargs["device"] = "rbln"
                except:
                    logger.warning("vLLM-RBLN device parameter not supported, using default")
            
            self._llm_engine = LLM(**llm_kwargs)
            self._model_name = model_name
            
            logger.info(f"vLLM-RBLN engine created for model: {model_name}")
            return self._llm_engine
            
        except Exception as e:
            logger.error(f"Failed to create vLLM-RBLN engine: {e}")
            raise
    
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute inference with vLLM-RBLN"""
        from vllm import SamplingParams
        
        # Extract proxy metadata
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)
        
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
        
        # Prepare prompts
        prompts = self._prepare_inputs(inputs, original_model, **kwargs)
        
        # Execute inference
        outputs = model.generate(prompts, sampling_params)
        
        # Convert output format if needed
        if proxy_method == "generate":
            return self._convert_to_hf_format(outputs, inputs, original_model, **kwargs)

        return outputs

    def _prepare_inputs(self, inputs: Any, original_model: Any = None, **kwargs) -> list:
        """Prepare inputs for vLLM"""
        if inputs is None:
            inputs = kwargs.get("input_ids", kwargs.get("inputs", None))
            if inputs is None:
                raise ValueError("No inputs provided")
        
        if isinstance(inputs, str):
            return [inputs]
        
        if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
            return inputs
        
        # Handle token IDs
        if hasattr(inputs, "input_ids"):
            token_ids = inputs.input_ids
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            token_ids = inputs["input_ids"]
        else:
            token_ids = inputs
        
        try:
            import torch
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            tokenizer = self._get_tokenizer(original_model)
            if tokenizer is not None:
                if isinstance(token_ids[0], list):
                    return [tokenizer.decode(ids, skip_special_tokens=False) for ids in token_ids]
                else:
                    return [tokenizer.decode(token_ids, skip_special_tokens=False)]
        except Exception as e:
            logger.warning(f"Failed to decode token IDs: {e}")
        
        return [str(inputs)]
    
    def _get_tokenizer(self, original_model: Any = None):
        """Get tokenizer"""
        tokenizer = None
        
        if original_model is not None:
            if hasattr(original_model, "tokenizer"):
                tokenizer = original_model.tokenizer
            elif hasattr(original_model, "config"):
                model_name = getattr(original_model.config, "name_or_path", None)
                if model_name:
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    except Exception:
                        pass
        
        if tokenizer is None and self._model_name:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            except Exception:
                pass
        
        return tokenizer
    
    def _convert_to_hf_format(self, vllm_outputs: Any, original_inputs: Any, 
                              original_model: Any = None, **kwargs) -> Any:
        """Convert vLLM outputs to HuggingFace format"""
        if isinstance(original_inputs, str):
            return vllm_outputs
        
        try:
            import torch
            tokenizer = self._get_tokenizer(original_model)
            if tokenizer is None:
                return vllm_outputs
            
            all_token_ids = []
            for output in vllm_outputs:
                prompt_token_ids = output.prompt_token_ids
                for completion in output.outputs:
                    generated_token_ids = list(prompt_token_ids) + list(completion.token_ids)
                    all_token_ids.append(generated_token_ids)
            
            if len(all_token_ids) == 1:
                return torch.tensor([all_token_ids[0]])
            else:
                max_len = max(len(ids) for ids in all_token_ids)
                pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                padded = [ids + [pad_token_id] * (max_len - len(ids)) for ids in all_token_ids]
                return torch.tensor(padded)
        except Exception as e:
            logger.warning(f"Failed to convert to HF format: {e}")
            return vllm_outputs
    
    def get_device_info(self) -> DeviceInfo:
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_id=self.device_id,
            name="Rebellions ATOM (vLLM)",
                    vendor="Rebellions",
        )
    
    def cleanup(self) -> None:
        self._llm_engine = None
        self._model_name = None
        super().cleanup()


class RBLNOptimumBackend(Backend):
    """
    Optimum-RBLN Backend Implementation
    
    HuggingFace model integration using optimum-rbln.
    Reference: https://docs.rbln.ai/latest/software/optimum/
    """
    
    def __init__(self, device_id: int = 0, **kwargs):
        super().__init__(device_id, **kwargs)
        self._compiled_model = None
        self._model_name: Optional[str] = None
        self._tokenizer = None
    
    @property
    def device_type(self) -> DeviceType:
        return DeviceType.NPU
    
    @property
    def name(self) -> str:
        return "rbln_optimum"
    
    def is_available(self) -> bool:
        try:
            from optimum.rbln import RBLNModelForCausalLM
            return True
        except ImportError:
            return False
    
    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.info(f"RBLN Optimum backend initialized on device {self.device_id}")
    
    def prepare_model(self, model: Any, model_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Prepare model using Optimum-RBLN
        
        Supports:
        - Direct model name/path loading
        - Converting HuggingFace models to RBLN format
        """
        config = model_config or {}
        
        # Extract model name
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "name_or_path"):
            model_name = model.name_or_path
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            model_name = model.config.name_or_path
        else:
            raise ValueError("Cannot determine model name/path")
        
        try:
            from optimum.rbln import RBLNModelForCausalLM
            from transformers import AutoTokenizer
            
            # Load or compile model for RBLN
            export = config.get("export", True)  # Export to RBLN format if needed
            
            if export:
                # Compile and export to RBLN format
                self._compiled_model = RBLNModelForCausalLM.from_pretrained(
                    model_name,
                    export=True,
                    rbln_device_id=self.device_id,
                    rbln_batch_size=config.get("batch_size", 1),
                    rbln_max_seq_len=config.get("max_seq_len", 2048),
                )
            else:
                # Load pre-compiled RBLN model
                self._compiled_model = RBLNModelForCausalLM.from_pretrained(
                    model_name,
                    rbln_device_id=self.device_id,
                )
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model_name = model_name
            
            logger.info(f"Optimum-RBLN model loaded: {model_name}")
            return self._compiled_model
            
        except Exception as e:
            logger.error(f"Failed to load Optimum-RBLN model: {e}")
            raise
    
    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """Execute inference with Optimum-RBLN"""
        import torch
        
        # Extract proxy metadata
        proxy_method = kwargs.pop("_proxy_method", None)
        original_model = kwargs.pop("original_model", None)
        
        # Prepare inputs
        if isinstance(inputs, str):
            # Tokenize string input
            inputs = self._tokenizer(inputs, return_tensors="pt")
        elif hasattr(inputs, "input_ids"):
            pass  # Already tokenized
        elif isinstance(inputs, dict) and "input_ids" in inputs:
            pass  # Already a dict with input_ids
        
        # Handle generate method
        if proxy_method == "generate":
            # Extract generation parameters
            max_new_tokens = kwargs.get("max_new_tokens", kwargs.get("max_length", 128))
            do_sample = kwargs.get("do_sample", False)
            temperature = kwargs.get("temperature", 1.0)
            top_p = kwargs.get("top_p", 1.0)
            
            # Generate using Optimum-RBLN model
            outputs = model.generate(
                **inputs if isinstance(inputs, dict) else {"input_ids": inputs.input_ids},
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            
            return outputs
        else:
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs if isinstance(inputs, dict) else {"input_ids": inputs.input_ids})
            return outputs
    
    def get_device_info(self) -> DeviceInfo:
        return DeviceInfo(
            device_type=DeviceType.NPU,
            device_id=self.device_id,
            name="Rebellions ATOM (Optimum)",
            vendor="Rebellions",
        )

    def cleanup(self) -> None:
        self._compiled_model = None
        self._tokenizer = None
        self._model_name = None
        super().cleanup()

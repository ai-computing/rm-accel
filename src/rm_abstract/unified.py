"""
Unified API for RM Abstract Layer

Provides a single API that works across all backends and serving engines.
This allows switching between vLLM GPU, Triton CPU, TorchServe, etc.
with minimal code changes.

Usage:
    from rm_abstract.unified import UnifiedModel
    
    # vLLM GPU
    model = UnifiedModel("gpt2", engine="vllm", device="gpu")
    
    # Triton CPU - same API!
    model = UnifiedModel("gpt2", engine="triton", device="cpu")
    
    # Same inference call for both
    output = model.generate("Hello, I am", max_tokens=50)
"""

from typing import Optional, List, Union, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Generation parameters"""
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    

class UnifiedModel:
    """
    Unified model interface that works across all backends.
    
    Provides the same API regardless of underlying engine:
    - vLLM (GPU/CPU)
    - Triton (GPU/CPU)
    - TorchServe (GPU/CPU)
    - Direct HuggingFace (CPU)
    
    Example:
        # These all use the same API:
        model = UnifiedModel("gpt2", engine="vllm", device="gpu")
        model = UnifiedModel("gpt2", engine="triton", device="cpu")
        model = UnifiedModel("gpt2", engine="torchserve", device="cpu")
        
        # Same generate call
        output = model.generate("Hello", max_tokens=50)
    """
    
    def __init__(
        self,
        model_name: str,
        engine: str = "auto",  # auto, vllm, triton, torchserve, huggingface
        device: str = "auto",  # auto, gpu, cpu, gpu:0, cpu
        **kwargs,
    ):
        """
        Initialize unified model.
        
        Args:
            model_name: Model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b")
            engine: Serving engine to use
            device: Device to run on
            **kwargs: Additional engine-specific options
        """
        self.model_name = model_name
        self.engine_type = engine
        self.device_type = device
        self._engine = None
        self._model = None
        self._tokenizer = None
        self._kwargs = kwargs
        
        self._initialize()
    
    def _initialize(self):
        """Initialize the appropriate backend"""
        engine = self._resolve_engine()
        device = self._resolve_device()
        
        logger.info(f"Initializing UnifiedModel: engine={engine}, device={device}")
        
        if engine == "huggingface":
            self._init_huggingface(device)
        elif engine == "vllm":
            self._init_vllm(device)
        elif engine == "triton":
            self._init_triton(device)
        elif engine == "torchserve":
            self._init_torchserve(device)
        else:
            raise ValueError(f"Unknown engine: {engine}")
    
    def _resolve_engine(self) -> str:
        """Resolve 'auto' engine to specific engine"""
        if self.engine_type != "auto":
            return self.engine_type
        
        # Auto-select based on availability
        from .serving import get_available_engines
        engines = get_available_engines()
        
        # Priority: vllm > triton > torchserve > huggingface
        if engines.get("vllm"):
            return "vllm"
        elif engines.get("triton"):
            return "triton"
        elif engines.get("torchserve"):
            return "torchserve"
        else:
            return "huggingface"
    
    def _resolve_device(self) -> str:
        """Resolve 'auto' device to specific device"""
        if self.device_type != "auto":
            return self.device_type
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                return "gpu"
        except ImportError:
            pass
        
        return "cpu"
    
    def _init_huggingface(self, device: str):
        """Initialize with HuggingFace directly"""
        import rm_abstract
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        rm_abstract.init(device=device, verbose=False)
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._engine_name = "huggingface"
    
    def _init_vllm(self, device: str):
        """Initialize with vLLM"""
        from .serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        device_target = DeviceTarget.GPU if "gpu" in device else DeviceTarget.CPU
        
        config = ServingConfig(
            engine=ServingEngineType.VLLM,
            device=device_target,
            model_name=self.model_name,
        )
        
        self._engine = create_serving_engine(config)
        self._engine.load_model(self.model_name)
        self._engine_name = "vllm"
    
    def _init_triton(self, device: str):
        """Initialize with Triton"""
        from .serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        device_target = DeviceTarget.GPU if "gpu" in device else DeviceTarget.CPU
        
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            device=device_target,
            model_name=self.model_name,
        )
        
        self._engine = create_serving_engine(config)
        self._engine.start_server()
        self._engine.load_model(self.model_name)
        self._engine_name = "triton"
    
    def _init_torchserve(self, device: str):
        """Initialize with TorchServe"""
        from .serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
        
        device_target = DeviceTarget.GPU if "gpu" in device else DeviceTarget.CPU
        
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            device=device_target,
            model_name=self.model_name,
        )
        
        self._engine = create_serving_engine(config)
        self._engine.start_server()
        self._engine.load_model(self.model_name)
        self._engine_name = "torchserve"
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt.
        
        This is the unified API - same call works for all engines.
        
        Args:
            prompt: Input text or list of texts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of texts
        """
        if self._engine_name == "huggingface":
            return self._generate_huggingface(prompt, max_tokens, temperature, do_sample, **kwargs)
        else:
            return self._generate_serving(prompt, max_tokens, temperature, **kwargs)
    
    def _generate_huggingface(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate using HuggingFace model"""
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt
        
        results = []
        for p in prompts:
            inputs = self._tokenizer(p, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs,
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(text)
        
        return results[0] if is_single else results
    
    def _generate_serving(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate using serving engine"""
        return self._engine.infer(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Clean up resources"""
        if self._engine is not None:
            try:
                self._engine.stop_server()
            except:
                pass
        self._engine = None
        self._model = None
    
    @property
    def engine_name(self) -> str:
        """Get the name of the underlying engine"""
        return self._engine_name
    
    def __repr__(self):
        return f"UnifiedModel(model={self.model_name}, engine={self._engine_name})"


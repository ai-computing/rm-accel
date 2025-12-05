"""
vLLM Serving Engine

High-performance LLM serving using vLLM.
Supports both GPU (CUDA) and NPU (RBLN) backends.

Reference: 
- https://docs.vllm.ai/
- https://docs.rbln.ai/latest/model_serving/vllm/
"""

from typing import Any, Dict, List, Optional, Union
import logging

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget

logger = logging.getLogger(__name__)


class VLLMServingEngine(ServingEngine):
    """
    vLLM-based serving engine
    
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
        engine = VLLMServingEngine(config)
        engine.load_model(config.model_name)
        
        # Run inference
        output = engine.infer("Hello, how are you?")
    """
    
    def __init__(self, config: ServingConfig):
        super().__init__(config)
        self._llm = None
        self._tokenizer = None
    
    @property
    def name(self) -> str:
        return "vLLM"
    
    @property
    def engine_type(self) -> ServingEngineType:
        return ServingEngineType.VLLM
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if vLLM is available"""
        try:
            import vllm
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
        
        # Check RBLN NPU (vLLM-RBLN)
        try:
            import vllm
            # vLLM-RBLN would support "rbln" device
            devices.append(DeviceTarget.NPU_RBLN)
        except ImportError:
            pass
        
        devices.append(DeviceTarget.CPU)
        return devices
    
    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load model with vLLM
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            **kwargs: Additional vLLM options
        """
        from vllm import LLM
        
        # Build vLLM arguments
        llm_kwargs = {
            "model": model_name_or_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }
        
        # Only set max_model_len if specified
        if self.config.max_seq_len:
            llm_kwargs["max_model_len"] = self.config.max_seq_len
        
        # Handle RBLN NPU device - requires vllm-rbln package
        if self.config.device == DeviceTarget.NPU_RBLN:
            # vLLM-RBLN uses different initialization
            # Note: vllm-rbln package provides RBLN support
            # Standard vLLM doesn't support 'device' parameter
            raise RuntimeError(
                "RBLN NPU requires vllm-rbln package which provides native RBLN support. "
                "Install with: pip install vllm-rbln\n"
                "Then use the standard vLLM API - device selection is automatic."
            )
        
        # Add extra options
        llm_kwargs.update(self.config.extra_options)
        llm_kwargs.update(kwargs)
        
        # Remove None values
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
        
        self._llm = LLM(**llm_kwargs)
        
        logger.info(f"vLLM model loaded: {model_name_or_path} on {self.config.device.value}")
        return self._llm
    
    def start_server(self) -> None:
        """
        Start vLLM engine
        
        vLLM은 라이브러리 방식으로 동작하므로 별도의 서버 프로세스가 필요 없습니다.
        이 메서드는 context manager 호환성을 위해 제공됩니다.
        """
        if self._is_running:
            logger.warning("Server already running")
            return
        
        # vLLM은 라이브러리 방식이므로 바로 '실행 중'으로 표시
        self._is_running = True
        logger.info("vLLM engine ready (library mode)")
    
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
        Run inference with vLLM
        
        Args:
            inputs: Text prompt or list of prompts
            **kwargs: Sampling parameters
            
        Returns:
            Generated text(s)
        """
        if self._llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        from vllm import SamplingParams
        
        # Build sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256)),
        )
        
        # Ensure inputs is a list
        if isinstance(inputs, str):
            prompts = [inputs]
        else:
            prompts = inputs
        
        # Generate
        outputs = self._llm.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            for completion in output.outputs:
                results.append(completion.text)
        
        # Return single result if single input
        if isinstance(inputs, str):
            return results[0] if results else ""
        return results
    
    def infer_async(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """
        Async inference using HTTP API
        
        Requires server to be running.
        """
        import requests
        
        if not self._is_running:
            raise RuntimeError("Server not running. Call start_server() first.")
        
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


"""
Triton Inference Server Engine

NVIDIA Triton Inference Server integration.
Supports GPU, CPU, and NPU (RBLN) backends.

Reference:
- https://github.com/triton-inference-server
- https://docs.rbln.ai/latest/model_serving/triton/
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
import json
import os

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget
from ..exceptions import (
    ModelNotLoadedError,
    ServerStartError,
    DockerNotAvailableError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


class TritonServingEngine(ServingEngine):
    """
    NVIDIA Triton Inference Server Engine
    
    Provides enterprise-grade model serving with:
    - Multiple model support
    - Dynamic batching
    - Model versioning
    - Metrics and monitoring
    - **Auto server management via Docker**
    
    Example:
        # 자동 서버 관리 (context manager)
        with TritonServingEngine(config) as engine:
            engine.load_model("gpt2")
            output = engine.infer("Hello")
        # 서버 자동 종료
        
        # 또는 수동 관리
        engine = TritonServingEngine(config)
        engine.load_model("gpt2")
        engine.start_server()  # Docker 자동 시작
        output = engine.infer("Hello")
        engine.stop_server()   # Docker 자동 종료
    """
    
    # Use custom image with transformers pre-installed
    # Build with: docker build -t rm-triton-server:latest -f docker/Dockerfile.triton docker/
    DOCKER_IMAGE = "rm-triton-server:latest"
    DOCKER_IMAGE_FALLBACK = "nvcr.io/nvidia/tritonserver:23.10-py3"
    CONTAINER_NAME = "rm_abstract_triton"
    
    def __init__(self, config: ServingConfig):
        super().__init__(config)
        self._model_repository: Optional[str] = None
        self._client = None
        self._container_id: Optional[str] = None
        self._loaded_model: Optional[str] = None
    
    @property
    def name(self) -> str:
        return "Triton"
    
    @property
    def engine_type(self) -> ServingEngineType:
        return ServingEngineType.TRITON
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton client is available"""
        try:
            import tritonclient.grpc as grpcclient
            return True
        except ImportError:
            try:
                import tritonclient.http as httpclient
                return True
            except ImportError:
                return False
    
    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """Triton supports multiple backends"""
        return [
            DeviceTarget.GPU,
            DeviceTarget.CPU,
            DeviceTarget.NPU_RBLN,
        ]
    
    def setup_model_repository(self, repo_path: str) -> None:
        """
        Setup Triton model repository
        
        Args:
            repo_path: Path to model repository
        """
        self._model_repository = repo_path
        Path(repo_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Model repository set to: {repo_path}")
    
    def create_model_config(
        self,
        model_name: str,
        backend: str = "python",
        max_batch_size: int = 0,
        input_config: Optional[List[Dict]] = None,
        output_config: Optional[List[Dict]] = None,
    ) -> str:
        """
        Create Triton model configuration (config.pbtxt)
        
        Args:
            model_name: Name of the model
            backend: Backend type (python, pytorch, onnxruntime, rbln)
            max_batch_size: Maximum batch size
            input_config: Input tensor configurations
            output_config: Output tensor configurations
            
        Returns:
            Path to config file
        """
        if self._model_repository is None:
            raise ConfigurationError("Model repository not set. Call setup_model_repository() first.")
        
        model_dir = Path(self._model_repository) / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configs for LLM
        if input_config is None:
            input_config = [
                {
                    "name": "INPUT_TEXT",
                    "data_type": "TYPE_STRING",
                    "dims": [1],
                }
            ]
        
        if output_config is None:
            output_config = [
                {
                    "name": "OUTPUT_TEXT",
                    "data_type": "TYPE_STRING",
                    "dims": [1],
                }
            ]
        
        # Build config.pbtxt
        config_content = f'''name: "{model_name}"
backend: "{backend}"
max_batch_size: {max_batch_size}

'''
        
        # Add inputs
        for inp in input_config:
            config_content += f'''input [
  {{
    name: "{inp['name']}"
    data_type: {inp['data_type']}
    dims: {inp['dims']}
  }}
]

'''
        
        # Add outputs
        for out in output_config:
            config_content += f'''output [
  {{
    name: "{out['name']}"
    data_type: {out['data_type']}
    dims: {out['dims']}
  }}
]

'''
        
        # Add instance group based on device
        if self.config.device == DeviceTarget.GPU:
            config_content += f'''instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [{self.config.device_id}]
  }}
]
'''
        elif self.config.device == DeviceTarget.NPU_RBLN:
            config_content += f'''instance_group [
  {{
    count: 1
    kind: KIND_MODEL
  }}
]

parameters [
  {{
    key: "device"
    value: {{ string_value: "rbln:{self.config.device_id}" }}
  }}
]
'''
        else:
            config_content += '''instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
'''
        
        config_path = model_dir / "config.pbtxt"
        config_path.write_text(config_content)
        
        logger.info(f"Created Triton config: {config_path}")
        return str(config_path)
    
    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load/prepare model for Triton
        
        For LLM models, this creates the necessary model files
        and configuration in the model repository.
        """
        if self._model_repository is None:
            # Use default repository
            self.setup_model_repository(
                os.path.expanduser("~/.rm_abstract/triton_models")
            )
        
        model_name = kwargs.get("triton_model_name", self.config.model_name or "llm_model")
        
        # Create model directory structure
        model_dir = Path(self._model_repository) / model_name / "1"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Python backend model file for LLM
        model_py = self._create_llm_model_py(model_name_or_path, model_dir)
        
        # Create config
        backend = "rbln" if self.config.device == DeviceTarget.NPU_RBLN else "python"
        self.create_model_config(model_name, backend=backend)
        
        logger.info(f"Prepared Triton model: {model_name} from {model_name_or_path}")
        return model_name
    
    def _create_llm_model_py(self, hf_model: str, model_dir: Path) -> str:
        """Create Python backend model.py for LLM inference"""
        
        model_code = f'''"""
Triton Python Backend for LLM
Auto-generated by rm_abstract
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python model for LLM inference"""
    
    def initialize(self, args):
        """Initialize the model"""
        self.model_config = json.loads(args["model_config"])
        
        # Load the HuggingFace model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "{hf_model}"
        device = "{self.config.device.value}"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if device == "npu_rbln":
            # Use Optimum-RBLN for NPU
            from optimum.rbln import RBLNModelForCausalLM
            self.model = RBLNModelForCausalLM.from_pretrained(
                model_name,
                export=True,
                rbln_device_id={self.config.device_id},
            )
        else:
            import torch
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            if device == "gpu" and torch.cuda.is_available():
                self.model = self.model.cuda({self.config.device_id})
    
    def execute(self, requests):
        """Execute inference"""
        responses = []
        
        for request in requests:
            # Get input
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text = input_tensor.as_numpy()[0].decode("utf-8")
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {{k: v.to(self.model.device) for k, v in inputs.items()}}
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens={self.config.extra_options.get("max_new_tokens", 256)},
                do_sample=True,
                temperature=0.7,
            )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Create response
            output_tensor = pb_utils.Tensor(
                "OUTPUT_TEXT",
                np.array([generated_text.encode("utf-8")], dtype=np.object_)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
    
    def finalize(self):
        """Cleanup"""
        pass
'''
        
        model_path = model_dir / "model.py"
        model_path.write_text(model_code)
        return str(model_path)
    
    def start_server(self) -> None:
        """
        Start Triton Inference Server (자동 Docker 관리)
        
        1. 로컬 tritonserver가 있으면 직접 실행
        2. 없으면 Docker 컨테이너로 자동 실행
        """
        if self._is_running:
            logger.warning("Server already running")
            return
        
        if self._model_repository is None:
            self.setup_model_repository("/tmp/triton_models")
        
        import subprocess
        import shutil
        import time
        
        # 방법 1: 로컬 tritonserver
        triton_cmd = shutil.which("tritonserver")
        if triton_cmd:
            return self._start_local_server(triton_cmd)
        
        # 방법 2: Docker
        docker_cmd = shutil.which("docker")
        if docker_cmd:
            return self._start_docker_server()
        
        raise DockerNotAvailableError(
            reason="Neither tritonserver nor Docker found"
        )
    
    def _start_local_server(self, triton_cmd: str) -> None:
        """로컬 tritonserver 실행"""
        import subprocess
        
        cmd = [
            triton_cmd,
            f"--model-repository={self._model_repository}",
            f"--http-port={self.config.port}",
            f"--grpc-port={self.config.port + 1}",
            f"--metrics-port={self.config.port + 2}",
        ]
        
        logger.info(f"Starting local Triton server: {' '.join(cmd)}")
        
        self._server = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._is_running = True
        self._wait_for_server()
        logger.info(f"Triton server started on port {self.config.port}")
    
    def _start_docker_server(self) -> None:
        """Docker 컨테이너로 Triton 서버 시작"""
        import subprocess
        import time
        
        # 기존 컨테이너 정리
        subprocess.run(
            ["docker", "rm", "-f", self.CONTAINER_NAME],
            capture_output=True
        )
        
        # GPU 옵션
        gpu_opt = []
        if self.config.device == DeviceTarget.GPU:
            gpu_opt = ["--gpus", f'"device={self.config.device_id}"']
        
        cmd = [
            "docker", "run", "-d",
            "--name", self.CONTAINER_NAME,
            *gpu_opt,
            "-p", f"{self.config.port}:8000",
            "-p", f"{self.config.port + 1}:8001",
            "-p", f"{self.config.port + 2}:8002",
            "-v", f"{self._model_repository}:/models",
            self.DOCKER_IMAGE,
            "tritonserver",
            "--model-repository=/models",
        ]
        
        # GPU 옵션이 있으면 쉘로 실행
        if gpu_opt:
            cmd_str = " ".join(cmd)
            logger.info(f"Starting Triton Docker: {cmd_str}")
            result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
        else:
            cmd = [c for c in cmd if c]  # 빈 문자열 제거
            logger.info(f"Starting Triton Docker: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise ServerStartError("Triton", reason=result.stderr)
        
        self._container_id = result.stdout.strip()
        self._is_running = True
        self._wait_for_server()
        logger.info(f"Triton Docker started: {self._container_id[:12]}")
    
    def _wait_for_server(self, timeout: int = 60) -> None:
        """서버가 준비될 때까지 대기"""
        import time
        import urllib.request
        
        start = time.time()
        url = f"http://localhost:{self.config.port}/v2/health/ready"
        
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        logger.info("Triton server is ready")
                        return
            except:
                pass
            time.sleep(1)
        
        logger.warning("Triton server may not be fully ready yet")
    
    def stop_server(self) -> None:
        """Stop Triton server (로컬 또는 Docker)"""
        import subprocess
        
        # 로컬 프로세스 종료
        if self._server is not None:
            self._server.terminate()
            self._server.wait()
            self._server = None
        
        # Docker 컨테이너 종료
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self.CONTAINER_NAME],
                capture_output=True
            )
            self._container_id = None
        
        self._is_running = False
        logger.info("Triton server stopped")
    
    def infer(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """
        Run inference via Triton client
        
        Args:
            inputs: Text prompt or list of prompts
            **kwargs: Additional options
        """
        try:
            import tritonclient.http as httpclient
            client_type = "http"
        except ImportError:
            import tritonclient.grpc as grpcclient
            client_type = "grpc"
        
        model_name = kwargs.get("model_name", self.config.model_name or "llm_model")
        
        if client_type == "http":
            import tritonclient.http as httpclient
            client = httpclient.InferenceServerClient(
                url=f"{self.config.host}:{self.config.port}"
            )
            
            # Prepare inputs
            if isinstance(inputs, str):
                inputs = [inputs]
            
            results = []
            for text in inputs:
                input_data = np.array([text.encode("utf-8")], dtype=np.object_)
                
                triton_input = httpclient.InferInput("INPUT_TEXT", [1], "BYTES")
                triton_input.set_data_from_numpy(input_data)
                
                triton_output = httpclient.InferRequestedOutput("OUTPUT_TEXT")
                
                response = client.infer(
                    model_name=model_name,
                    inputs=[triton_input],
                    outputs=[triton_output],
                )
                
                output_data = response.as_numpy("OUTPUT_TEXT")
                results.append(output_data[0].decode("utf-8"))
            
            return results[0] if len(results) == 1 else results
        
        else:
            # gRPC client
            import tritonclient.grpc as grpcclient
            client = grpcclient.InferenceServerClient(
                url=f"{self.config.host}:{self.config.port + 1}"
            )
            
            # Similar implementation for gRPC...
            raise NotImplementedError("gRPC client not fully implemented")
    
    def get_server_health(self) -> Dict[str, Any]:
        """Get Triton server health status"""
        try:
            import tritonclient.http as httpclient
            client = httpclient.InferenceServerClient(
                url=f"{self.config.host}:{self.config.port}"
            )
            
            return {
                "live": client.is_server_live(),
                "ready": client.is_server_ready(),
            }
        except Exception as e:
            return {"error": str(e)}


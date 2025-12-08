"""
Triton Inference Server Engine

Model serving using NVIDIA Triton Inference Server.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging
import subprocess
import time

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget

logger = logging.getLogger(__name__)


@dataclass
class TritonConfig:
    """Triton-specific configuration"""

    # Docker configuration
    docker_image: str = "nvcr.io/nvidia/tritonserver:23.10-py3"
    container_name: str = "rm_abstract_triton"

    # Model repository
    model_repository: str = "/models"

    # Server ports
    http_port: int = 8000
    grpc_port: int = 8001
    metrics_port: int = 8002

    # GPU configuration
    gpus: str = "all"  # "all" or specific GPU IDs like "0,1"


class TritonEngine(ServingEngine):
    """
    Triton Inference Server engine.

    Uses Docker to manage Triton server lifecycle.

    Example:
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            model_name="gpt2",
            port=8000,
        )
        engine = TritonEngine(config)
        engine.start_server()
        output = engine.infer("Hello!")
        engine.stop_server()
    """

    ENGINE_TYPE = ServingEngineType.TRITON
    ENGINE_NAME = "Triton Inference Server"

    def __init__(
        self,
        config: ServingConfig,
        triton_config: Optional[TritonConfig] = None,
    ):
        super().__init__(config)
        self.triton_config = triton_config or TritonConfig()
        self._container_id: Optional[str] = None

    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton is available (Docker installed)"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """Triton supports GPU and CPU"""
        return [DeviceTarget.GPU, DeviceTarget.CPU]

    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load model into Triton.

        For Triton, this prepares the model repository.
        """
        self.config.model_name = model_name_or_path
        logger.info(f"Model configured for Triton: {model_name_or_path}")
        return model_name_or_path

    def start_server(self) -> None:
        """Start Triton server using Docker"""
        if self._is_running:
            logger.warning("Triton server already running")
            return

        # Build Docker run command
        cmd = [
            "docker", "run", "-d",
            "--name", self.triton_config.container_name,
            "-p", f"{self.triton_config.http_port}:8000",
            "-p", f"{self.triton_config.grpc_port}:8001",
            "-p", f"{self.triton_config.metrics_port}:8002",
        ]

        # Add GPU support if available
        if self.config.device == DeviceTarget.GPU:
            cmd.extend(["--gpus", self.triton_config.gpus])

        # Add model repository volume
        cmd.extend([
            "-v", f"{self.triton_config.model_repository}:/models",
            self.triton_config.docker_image,
            "tritonserver",
            "--model-repository=/models",
        ])

        try:
            # Remove existing container if exists
            subprocess.run(
                ["docker", "rm", "-f", self.triton_config.container_name],
                capture_output=True,
            )

            # Start new container
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start Triton: {result.stderr}")

            self._container_id = result.stdout.strip()
            self._is_running = True

            # Wait for server to be ready
            self._wait_for_ready()
            logger.info("Triton server started")

        except Exception as e:
            logger.error(f"Failed to start Triton server: {e}")
            raise

    def stop_server(self) -> None:
        """Stop Triton server"""
        if not self._is_running:
            return

        try:
            subprocess.run(
                ["docker", "stop", self.triton_config.container_name],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", self.triton_config.container_name],
                capture_output=True,
            )
        except Exception as e:
            logger.error(f"Error stopping Triton: {e}")

        self._is_running = False
        self._container_id = None
        logger.info("Triton server stopped")

    def _wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for Triton to be ready"""
        import requests

        url = f"http://localhost:{self.triton_config.http_port}/v2/health/ready"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise TimeoutError("Triton server did not become ready in time")

    def infer(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference with Triton.

        Args:
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        if not self._is_running:
            raise RuntimeError("Triton server not running")

        try:
            import tritonclient.http as httpclient
        except ImportError:
            raise ImportError(
                "tritonclient not installed. "
                "Install with: pip install tritonclient[http]"
            )

        # Create client
        client = httpclient.InferenceServerClient(
            url=f"localhost:{self.triton_config.http_port}"
        )

        # Prepare inputs (simplified - actual implementation depends on model)
        import numpy as np

        if isinstance(inputs, str):
            inputs = np.array([[inputs]], dtype=object)

        infer_input = httpclient.InferInput(
            "INPUT", inputs.shape, "BYTES"
        )
        infer_input.set_data_from_numpy(inputs)

        # Run inference
        result = client.infer(
            model_name=self.config.model_name,
            inputs=[infer_input],
        )

        return result.as_numpy("OUTPUT")

    def health_check(self) -> bool:
        """Check Triton health"""
        if not self._is_running:
            return False

        try:
            import requests

            response = requests.get(
                f"http://localhost:{self.triton_config.http_port}/v2/health/ready",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get Triton metrics"""
        metrics = super().get_metrics()

        if self._is_running:
            try:
                import requests

                response = requests.get(
                    f"http://localhost:{self.triton_config.metrics_port}/metrics",
                    timeout=5,
                )
                if response.status_code == 200:
                    metrics["prometheus_metrics"] = response.text
            except Exception:
                pass

        return metrics

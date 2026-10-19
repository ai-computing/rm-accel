"""
Triton Inference Server Engine

Model serving using NVIDIA Triton Inference Server.
Supports automatic model conversion and Docker container management.
"""

import atexit
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import DeviceTarget, ServingConfig, ServingEngine, ServingEngineType

logger = logging.getLogger(__name__)


@dataclass
class TritonConfig:
    """Triton-specific configuration"""

    # Docker configuration
    docker_image: str = "nvcr.io/nvidia/tritonserver:24.01-py3"
    container_name: str = "rm_abstract_triton"

    # Model repository
    model_repository: Optional[str] = None  # Auto-created if None

    # Server ports
    http_port: int = 8000
    grpc_port: int = 8001
    metrics_port: int = 8002

    # GPU configuration
    gpus: str = "all"  # "all" or specific GPU IDs like "0,1"

    # Auto management
    auto_start: bool = True
    auto_convert: bool = True
    cleanup_on_exit: bool = True


class TritonEngine(ServingEngine):
    """
    Triton Inference Server engine with automatic setup.

    Features:
    - Automatic Docker container management
    - Automatic HuggingFace to ONNX conversion
    - Automatic model repository creation

    Example:
        config = ServingConfig(
            engine=ServingEngineType.TRITON,
            model_name="gpt2",
        )
        engine = TritonEngine(config)
        engine.load_model("gpt2")  # Auto-converts and starts server
        output = engine.infer("Hello!")
    """

    ENGINE_TYPE = ServingEngineType.TRITON
    ENGINE_NAME = "Triton Inference Server"

    # Class-level tracking of running containers
    _running_containers: Dict[str, "TritonEngine"] = {}

    def __init__(
        self,
        config: ServingConfig,
        triton_config: Optional[TritonConfig] = None,
    ):
        super().__init__(config)
        self.triton_config = triton_config or TritonConfig()
        self._container_id: Optional[str] = None
        self._model_repo_path: Optional[Path] = None
        self._tokenizer = None

        # Set up auto cleanup
        if self.triton_config.cleanup_on_exit:
            atexit.register(self._cleanup)

    @classmethod
    def is_available(cls) -> bool:
        """Check if Docker is available for Triton"""
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
    def is_server_running(cls, port: int = 8000) -> bool:
        """Check if Triton server is already running"""
        try:
            import requests

            response = requests.get(
                f"http://localhost:{port}/v2/health/ready", timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """Triton supports GPU and CPU"""
        return [DeviceTarget.GPU, DeviceTarget.CPU]

    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load and prepare model for Triton.

        This will:
        1. Convert model to ONNX if needed
        2. Create Triton model repository
        3. Start Triton Docker container
        4. Wait for model to be ready
        """
        self.config.model_name = model_name_or_path

        if self.triton_config.auto_convert:
            self._prepare_model(model_name_or_path)

        if self.triton_config.auto_start and not self._is_running:
            self.start_server()

        logger.info(f"Model loaded for Triton: {model_name_or_path}")
        return model_name_or_path

    def _prepare_model(self, model_name: str) -> None:
        """Convert model and create Triton repository."""
        from .model_converter import ModelConverter

        converter = ModelConverter()

        # Convert to ONNX
        logger.info(f"Converting {model_name} to ONNX...")
        onnx_path = converter.to_onnx(
            model_name,
            device="cuda" if self.config.device == DeviceTarget.GPU else "cpu",
        )

        # Create model repository
        cache_dir = Path(os.path.expanduser("~/.rm_abstract"))
        self._model_repo_path = cache_dir / "triton_models"
        self._model_repo_path.mkdir(parents=True, exist_ok=True)

        converter.create_triton_config(
            model_name=model_name,
            onnx_path=onnx_path,
            output_dir=str(self._model_repo_path),
            device="gpu" if self.config.device == DeviceTarget.GPU else "cpu",
        )

        self.triton_config.model_repository = str(self._model_repo_path)

        # Load tokenizer for inference
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def start_server(self) -> None:
        """Start Triton server using Docker"""
        if self._is_running:
            logger.warning("Triton server already running")
            return

        if self.is_server_running(self.triton_config.http_port):
            logger.info("Triton server already running on port, reusing")
            self._is_running = True
            return

        if not self.is_available():
            raise RuntimeError("Docker is not available")

        # Build Docker run command
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.triton_config.container_name,
            "-p",
            f"{self.triton_config.http_port}:8000",
            "-p",
            f"{self.triton_config.grpc_port}:8001",
            "-p",
            f"{self.triton_config.metrics_port}:8002",
            "--shm-size=1g",
        ]

        # Add GPU support if available
        if self.config.device == DeviceTarget.GPU:
            cmd.extend(["--gpus", self.triton_config.gpus])

        # Add model repository volume
        model_repo = self.triton_config.model_repository or "/models"
        cmd.extend(
            [
                "-v",
                f"{model_repo}:/models",
                self.triton_config.docker_image,
                "tritonserver",
                "--model-repository=/models",
                "--strict-model-config=false",  # Allow auto-completion
            ]
        )

        try:
            # Remove existing container if exists
            subprocess.run(
                ["docker", "rm", "-f", self.triton_config.container_name],
                capture_output=True,
            )

            # Start new container
            logger.info("Starting Triton Docker container...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start Triton: {result.stderr}")

            self._container_id = result.stdout.strip()
            self._is_running = True

            # Track running container
            TritonEngine._running_containers[self.triton_config.container_name] = self

            # Wait for server to be ready
            self._wait_for_ready()
            logger.info("Triton server started successfully")

        except Exception as e:
            logger.error(f"Failed to start Triton server: {e}")
            self._cleanup()
            raise

    def stop_server(self) -> None:
        """Stop Triton server"""
        if not self._is_running:
            return

        try:
            logger.info("Stopping Triton server...")
            subprocess.run(
                ["docker", "stop", self.triton_config.container_name],
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", "-f", self.triton_config.container_name],
                capture_output=True,
            )
        except Exception as e:
            logger.error(f"Error stopping Triton: {e}")

        self._is_running = False
        self._container_id = None

        # Remove from tracking
        TritonEngine._running_containers.pop(self.triton_config.container_name, None)
        logger.info("Triton server stopped")

    def _cleanup(self) -> None:
        """Cleanup on exit"""
        if self._is_running and self.triton_config.cleanup_on_exit:
            self.stop_server()

    def _wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for Triton to be ready"""

        import requests

        url = f"http://localhost:{self.triton_config.http_port}/v2/health/ready"
        start_time = time.time()

        print("[Triton] Starting Docker container...", flush=True)
        check_interval = 2
        dots = 0

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Check if container is still running
            if self._container_id:
                result = subprocess.run(
                    ["docker", "ps", "-q", "-f", f"id={self._container_id}"],
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    # Container stopped, get logs
                    logs = subprocess.run(
                        ["docker", "logs", self.triton_config.container_name],
                        capture_output=True,
                        text=True,
                    )
                    print("\n[Triton] Container failed!", flush=True)
                    raise RuntimeError(f"Triton container exited: {logs.stderr}")

            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"\n[Triton] Server ready! ({elapsed}s)", flush=True)
                    logger.info("Triton server is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Progress indicator
            dots = (dots + 1) % 4
            progress = "." * (dots + 1)
            print(
                f"\r[Triton] Waiting for server{progress.ljust(4)} ({elapsed}/{timeout}s)",
                end="",
                flush=True,
            )
            time.sleep(check_interval)

        print("\n[Triton] Timeout!", flush=True)
        raise TimeoutError(f"Triton server did not become ready within {timeout}s")

    def infer(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference with Triton.

        Args:
            inputs: Text input or tokenized input
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            Generated text or logits
        """
        if not self._is_running:
            raise RuntimeError("Triton server not running. Call load_model() first.")

        try:
            import numpy as np
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

        safe_name = self.config.model_name.replace("/", "_")

        # Tokenize if string input
        if isinstance(inputs, str):
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

            tokens = self._tokenizer(
                inputs,
                return_tensors="np",
                padding=True,
                max_length=512,
                truncation=True,
            )
            input_ids = tokens["input_ids"].astype(np.int64)
            attention_mask = tokens["attention_mask"].astype(np.int64)
        else:
            input_ids = np.array(inputs, dtype=np.int64)
            attention_mask = np.ones_like(input_ids)

        # Prepare Triton inputs
        triton_inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
        ]
        triton_inputs[0].set_data_from_numpy(input_ids)
        triton_inputs[1].set_data_from_numpy(attention_mask)

        # Run inference
        result = client.infer(
            model_name=safe_name,
            inputs=triton_inputs,
        )

        logits = result.as_numpy("logits")

        # Simple greedy decoding for text generation
        max_tokens = kwargs.get("max_tokens", 50)
        generated_ids = input_ids.tolist()[0]

        for _ in range(max_tokens):
            # Get next token
            next_token_logits = logits[0, -1, :]
            next_token = int(np.argmax(next_token_logits))

            if next_token == self._tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

            # Update inputs for next iteration
            input_ids = np.array([generated_ids], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)

            triton_inputs[0].set_data_from_numpy(input_ids)
            triton_inputs[1].set_data_from_numpy(attention_mask)

            result = client.infer(model_name=safe_name, inputs=triton_inputs)
            logits = result.as_numpy("logits")

        # Decode output
        generated_text = self._tokenizer.decode(
            generated_ids[len(tokens["input_ids"][0]) :], skip_special_tokens=True
        )

        return {"generated_text": generated_text}

    def health_check(self) -> bool:
        """Check Triton health"""
        return self.is_server_running(self.triton_config.http_port)

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

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

"""
TorchServe Engine

Model serving using PyTorch TorchServe.
Supports automatic MAR packaging and server management.
"""

import os
import atexit
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging
import subprocess
import time
from pathlib import Path
import tempfile

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget

logger = logging.getLogger(__name__)


@dataclass
class TorchServeConfig:
    """TorchServe-specific configuration"""

    # Server ports
    inference_port: int = 8080
    management_port: int = 8081
    metrics_port: int = 8082

    # Model store
    model_store: Optional[str] = None  # Auto-created if None

    # Auto management
    auto_start: bool = True
    auto_package: bool = True
    cleanup_on_exit: bool = True

    # Worker configuration
    initial_workers: int = 1
    max_workers: int = 4


class TorchServeEngine(ServingEngine):
    """
    TorchServe engine with automatic setup.

    Features:
    - Automatic MAR (Model Archive) packaging
    - Automatic server lifecycle management
    - Dynamic model registration

    Example:
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            model_name="gpt2",
        )
        engine = TorchServeEngine(config)
        engine.load_model("gpt2")  # Auto-packages and starts server
        output = engine.infer("Hello!")
    """

    ENGINE_TYPE = ServingEngineType.TORCHSERVE
    ENGINE_NAME = "TorchServe"

    # Class-level tracking of running servers
    _running_servers: Dict[str, "TorchServeEngine"] = {}

    def __init__(
        self,
        config: ServingConfig,
        ts_config: Optional[TorchServeConfig] = None,
    ):
        super().__init__(config)
        self.ts_config = ts_config or TorchServeConfig()
        self._process: Optional[subprocess.Popen] = None
        self._model_store_path: Optional[Path] = None
        self._mar_file: Optional[str] = None
        self._tokenizer = None
        self._config_file: Optional[Path] = None

        # Set up auto cleanup
        if self.ts_config.cleanup_on_exit:
            atexit.register(self._cleanup)

    @classmethod
    def is_available(cls) -> bool:
        """Check if TorchServe is available"""
        try:
            result = subprocess.run(
                ["torchserve", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @classmethod
    def is_server_running(cls, port: int = 8080) -> bool:
        """Check if TorchServe server is already running"""
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/ping", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def supported_devices(cls) -> List[DeviceTarget]:
        """TorchServe supports GPU and CPU"""
        return [DeviceTarget.GPU, DeviceTarget.CPU]

    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load and prepare model for TorchServe.

        This will:
        1. Convert model to MAR format if needed
        2. Start TorchServe server
        3. Register the model
        4. Wait for model to be ready
        """
        self.config.model_name = model_name_or_path

        if self.ts_config.auto_package:
            self._prepare_model(model_name_or_path)

        if self.ts_config.auto_start and not self._is_running:
            self.start_server()

        # Register model if server is running
        if self._is_running and self._mar_file:
            self._register_model()

        logger.info(f"Model loaded for TorchServe: {model_name_or_path}")
        return model_name_or_path

    def _prepare_model(self, model_name: str) -> None:
        """Convert model to MAR format."""
        from .model_converter import ModelConverter

        converter = ModelConverter()

        # Convert to MAR
        logger.info(f"Converting {model_name} to MAR format...")
        mar_path = converter.to_mar(model_name)
        self._mar_file = mar_path

        # Set up model store
        cache_dir = Path(os.path.expanduser("~/.rm_abstract"))
        self._model_store_path = cache_dir / "torchserve_models"
        self._model_store_path.mkdir(parents=True, exist_ok=True)

        # Copy MAR to model store
        import shutil
        safe_name = model_name.replace("/", "_")
        dest_mar = self._model_store_path / f"{safe_name}.mar"
        if Path(mar_path).exists() and not dest_mar.exists():
            shutil.copy(mar_path, dest_mar)

        self.ts_config.model_store = str(self._model_store_path)
        self._mar_file = str(dest_mar)

        # Load tokenizer for inference
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def start_server(self) -> None:
        """Start TorchServe server"""
        if self._is_running:
            logger.warning("TorchServe already running")
            return

        if self.is_server_running(self.ts_config.inference_port):
            logger.info("TorchServe server already running on port, reusing")
            self._is_running = True
            return

        if not self.is_available():
            raise RuntimeError("TorchServe is not installed")

        # Create config file
        self._create_config_file()

        # Build command
        model_store = self.ts_config.model_store or str(self._model_store_path)
        cmd = [
            "torchserve",
            "--start",
            "--model-store", model_store,
            "--ts-config", str(self._config_file),
            "--ncs",  # No config snapshot
        ]

        try:
            # Stop any existing server first
            subprocess.run(
                ["torchserve", "--stop"],
                capture_output=True,
                timeout=10,
            )
            time.sleep(2)

            logger.info("Starting TorchServe server...")
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to be ready
            self._wait_for_ready()
            self._is_running = True

            # Track running server
            TorchServeEngine._running_servers[str(self.ts_config.inference_port)] = self
            logger.info("TorchServe server started successfully")

        except Exception as e:
            logger.error(f"Failed to start TorchServe: {e}")
            self._cleanup()
            raise

    def _create_config_file(self) -> None:
        """Create TorchServe config.properties file"""
        cache_dir = Path(os.path.expanduser("~/.rm_abstract"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._config_file = cache_dir / "torchserve_config.properties"

        config_content = f"""
inference_address=http://0.0.0.0:{self.ts_config.inference_port}
management_address=http://0.0.0.0:{self.ts_config.management_port}
metrics_address=http://0.0.0.0:{self.ts_config.metrics_port}
number_of_netty_threads=4
job_queue_size=10
model_store={self.ts_config.model_store or ''}
load_models=all
"""
        self._config_file.write_text(config_content.strip())

    def stop_server(self) -> None:
        """Stop TorchServe server"""
        if not self._is_running:
            return

        try:
            logger.info("Stopping TorchServe server...")
            subprocess.run(
                ["torchserve", "--stop"],
                capture_output=True,
                timeout=30,
            )
        except Exception as e:
            logger.error(f"Error stopping TorchServe: {e}")

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

        self._is_running = False

        # Remove from tracking
        TorchServeEngine._running_servers.pop(str(self.ts_config.inference_port), None)
        logger.info("TorchServe server stopped")

    def _cleanup(self) -> None:
        """Cleanup on exit"""
        if self._is_running and self.ts_config.cleanup_on_exit:
            self.stop_server()

    def _wait_for_ready(self, timeout: int = 30) -> None:
        """Wait for TorchServe to be ready"""
        import requests

        url = f"http://localhost:{self.ts_config.inference_port}/ping"
        start_time = time.time()

        print(f"[TorchServe] Starting server...", flush=True)
        check_interval = 1
        dots = 0

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            # Check if process is still running
            if self._process and self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else b""
                print(f"\n[TorchServe] Server failed!", flush=True)
                raise RuntimeError(f"TorchServe process exited: {stderr.decode()}")

            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print(f"\n[TorchServe] Server ready! ({elapsed}s)", flush=True)
                    logger.info("TorchServe server is ready")
                    return
            except requests.exceptions.RequestException:
                pass

            # Progress indicator
            dots = (dots + 1) % 4
            progress = "." * (dots + 1)
            print(f"\r[TorchServe] Waiting for server{progress.ljust(4)} ({elapsed}/{timeout}s)", end="", flush=True)
            time.sleep(check_interval)

        print(f"\n[TorchServe] Timeout!", flush=True)
        raise TimeoutError(f"TorchServe did not become ready within {timeout}s")

    def _register_model(self) -> None:
        """Register model with running TorchServe"""
        if not self._mar_file:
            return

        import requests

        safe_name = self.config.model_name.replace("/", "_")
        url = f"http://localhost:{self.ts_config.management_port}/models"

        # Check if model already registered
        try:
            response = requests.get(
                f"{url}/{safe_name}",
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(f"Model {safe_name} already registered")
                return
        except Exception:
            pass

        # Register model
        params = {
            "url": Path(self._mar_file).name,
            "model_name": safe_name,
            "initial_workers": self.ts_config.initial_workers,
        }

        try:
            response = requests.post(url, params=params, timeout=60)
            if response.status_code in [200, 202]:
                logger.info(f"Model {safe_name} registered successfully")
                # Wait for model to be loaded
                self._wait_for_model_ready(safe_name)
            else:
                logger.warning(f"Model registration returned: {response.text}")
        except Exception as e:
            logger.error(f"Failed to register model: {e}")

    def _wait_for_model_ready(self, model_name: str, timeout: int = 120) -> None:
        """Wait for model to be ready for inference"""
        import requests

        url = f"http://localhost:{self.ts_config.management_port}/models/{model_name}"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    workers = data.get("workers", [])
                    if workers and any(w.get("status") == "READY" for w in workers):
                        logger.info(f"Model {model_name} is ready")
                        return
            except Exception:
                pass
            time.sleep(2)

        logger.warning(f"Model may not be fully ready after {timeout}s")

    def infer(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference with TorchServe.

        Args:
            inputs: Text input or dict with 'data' key
            **kwargs: Additional options (max_tokens, temperature, etc.)

        Returns:
            Generated text or model output
        """
        if not self._is_running:
            raise RuntimeError("TorchServe not running. Call load_model() first.")

        import requests

        safe_name = self.config.model_name.replace("/", "_")
        url = f"http://localhost:{self.ts_config.inference_port}/predictions/{safe_name}"

        # Prepare input
        if isinstance(inputs, str):
            data = {"text": inputs}
        elif isinstance(inputs, dict):
            data = inputs
        else:
            data = {"data": inputs}

        # Add generation parameters
        data.update({
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
        })

        try:
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"TorchServe inference failed: {e}")

    def health_check(self) -> bool:
        """Check TorchServe health"""
        return self.is_server_running(self.ts_config.inference_port)

    def get_metrics(self) -> Dict[str, Any]:
        """Get TorchServe metrics"""
        metrics = super().get_metrics()

        if self._is_running:
            try:
                import requests
                response = requests.get(
                    f"http://localhost:{self.ts_config.metrics_port}/metrics",
                    timeout=5,
                )
                if response.status_code == 200:
                    metrics["prometheus_metrics"] = response.text
            except Exception:
                pass

        return metrics

    def register_model(self, model_name: str, mar_file: str) -> None:
        """Register a model with TorchServe"""
        import requests

        url = f"http://localhost:{self.ts_config.management_port}/models"
        params = {
            "url": mar_file,
            "model_name": model_name,
            "initial_workers": self.ts_config.initial_workers,
        }

        response = requests.post(url, params=params, timeout=60)
        response.raise_for_status()
        logger.info(f"Model {model_name} registered")

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model from TorchServe"""
        import requests

        url = f"http://localhost:{self.ts_config.management_port}/models/{model_name}"
        response = requests.delete(url, timeout=30)
        response.raise_for_status()
        logger.info(f"Model {model_name} unregistered")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

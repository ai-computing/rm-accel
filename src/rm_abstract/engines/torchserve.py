"""
TorchServe Engine

Model serving using PyTorch TorchServe.
"""

from typing import Any, Dict, List, Optional, Union
import logging
import subprocess
import time

from .base import ServingEngine, ServingEngineType, ServingConfig, DeviceTarget

logger = logging.getLogger(__name__)


class TorchServeEngine(ServingEngine):
    """
    TorchServe-based serving engine.

    Example:
        config = ServingConfig(
            engine=ServingEngineType.TORCHSERVE,
            model_name="gpt2",
            port=8080,
        )
        engine = TorchServeEngine(config)
        engine.load_model("gpt2")
        engine.start_server()
        output = engine.infer("Hello!")
    """

    ENGINE_TYPE = ServingEngineType.TORCHSERVE
    ENGINE_NAME = "TorchServe"

    def __init__(self, config: ServingConfig):
        super().__init__(config)
        self._model_store: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None

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
    def supported_devices(cls) -> List[DeviceTarget]:
        """TorchServe supports GPU and CPU"""
        return [DeviceTarget.GPU, DeviceTarget.CPU]

    def load_model(self, model_name_or_path: str, **kwargs) -> Any:
        """
        Load model for TorchServe.

        This archives the model into a .mar file for TorchServe.
        """
        self.config.model_name = model_name_or_path

        # For production, you'd create a .mar archive here
        logger.info(f"Model configured for TorchServe: {model_name_or_path}")
        return model_name_or_path

    def start_server(self) -> None:
        """Start TorchServe server"""
        if self._is_running:
            logger.warning("TorchServe already running")
            return

        cmd = [
            "torchserve",
            "--start",
            "--model-store", self._model_store or "model_store",
            "--ts-config", "config.properties",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to be ready
            self._wait_for_ready()
            self._is_running = True
            logger.info("TorchServe started")

        except Exception as e:
            logger.error(f"Failed to start TorchServe: {e}")
            raise

    def stop_server(self) -> None:
        """Stop TorchServe server"""
        if not self._is_running:
            return

        try:
            subprocess.run(["torchserve", "--stop"], capture_output=True)
        except Exception as e:
            logger.error(f"Error stopping TorchServe: {e}")

        if self._process:
            self._process.terminate()
            self._process = None

        self._is_running = False
        logger.info("TorchServe stopped")

    def _wait_for_ready(self, timeout: int = 60) -> None:
        """Wait for TorchServe to be ready"""
        import requests

        url = f"http://localhost:{self.config.port}/ping"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise TimeoutError("TorchServe did not become ready in time")

    def infer(self, inputs: Any, **kwargs) -> Any:
        """
        Run inference with TorchServe.

        Args:
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        if not self._is_running:
            raise RuntimeError("TorchServe not running")

        import requests

        url = f"http://localhost:{self.config.port}/predictions/{self.config.model_name}"

        if isinstance(inputs, str):
            # Text input
            response = requests.post(url, data=inputs)
        else:
            # JSON input
            response = requests.post(url, json=inputs)

        response.raise_for_status()
        return response.json()

    def health_check(self) -> bool:
        """Check TorchServe health"""
        if not self._is_running:
            return False

        try:
            import requests

            response = requests.get(
                f"http://localhost:{self.config.port}/ping",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def register_model(self, model_name: str, mar_file: str) -> None:
        """Register a model with TorchServe"""
        import requests

        url = f"http://localhost:{self.config.port}/models"
        params = {
            "url": mar_file,
            "model_name": model_name,
            "initial_workers": 1,
        }

        response = requests.post(url, params=params)
        response.raise_for_status()
        logger.info(f"Model {model_name} registered")

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model from TorchServe"""
        import requests

        url = f"http://localhost:{self.config.port}/models/{model_name}"
        response = requests.delete(url)
        response.raise_for_status()
        logger.info(f"Model {model_name} unregistered")

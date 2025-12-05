"""
Tests for custom exceptions.
"""

import pytest
from rm_abstract.exceptions import (
    RMAbstractError,
    InitializationError,
    NotInitializedError,
    BackendError,
    BackendNotAvailableError,
    BackendInitError,
    DeviceNotFoundError,
    ModelError,
    ModelLoadError,
    ModelNotLoadedError,
    InferenceError,
    ServingError,
    ServerStartError,
    ServerNotRunningError,
    ServerConnectionError,
    ConfigurationError,
    InvalidDeviceError,
    DockerError,
    DockerNotAvailableError,
    DockerImageNotFoundError,
    DependencyError,
    PackageNotInstalledError,
)


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy"""
    
    def test_base_exception(self):
        """RMAbstractError is base for all exceptions"""
        assert issubclass(InitializationError, RMAbstractError)
        assert issubclass(BackendError, RMAbstractError)
        assert issubclass(ModelError, RMAbstractError)
        assert issubclass(ServingError, RMAbstractError)
        assert issubclass(ConfigurationError, RMAbstractError)
        assert issubclass(DockerError, RMAbstractError)
        assert issubclass(DependencyError, RMAbstractError)
    
    def test_backend_subclasses(self):
        """Backend error subclasses"""
        assert issubclass(BackendNotAvailableError, BackendError)
        assert issubclass(BackendInitError, BackendError)
        assert issubclass(DeviceNotFoundError, BackendError)
    
    def test_model_subclasses(self):
        """Model error subclasses"""
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(ModelNotLoadedError, ModelError)
        assert issubclass(InferenceError, ModelError)
    
    def test_serving_subclasses(self):
        """Serving error subclasses"""
        assert issubclass(ServerStartError, ServingError)
        assert issubclass(ServerNotRunningError, ServingError)
        assert issubclass(ServerConnectionError, ServingError)


class TestNotInitializedError:
    """Test NotInitializedError"""
    
    def test_default_message(self):
        err = NotInitializedError()
        assert "not initialized" in str(err).lower()
        assert "init()" in str(err)
    
    def test_custom_message(self):
        err = NotInitializedError("Custom message")
        assert str(err) == "Custom message"


class TestBackendNotAvailableError:
    """Test BackendNotAvailableError"""
    
    def test_basic(self):
        err = BackendNotAvailableError("gpu")
        assert "gpu" in str(err)
        assert err.backend_name == "gpu"
    
    def test_with_reason(self):
        err = BackendNotAvailableError("gpu", reason="CUDA not found")
        assert "CUDA not found" in str(err)
        assert err.reason == "CUDA not found"
    
    def test_with_install_hint(self):
        err = BackendNotAvailableError("gpu", install_hint="pip install torch")
        assert "pip install torch" in str(err)
        assert err.install_hint == "pip install torch"


class TestDeviceNotFoundError:
    """Test DeviceNotFoundError"""
    
    def test_basic(self):
        err = DeviceNotFoundError("gpu:5")
        assert "gpu:5" in str(err)
        assert err.device == "gpu:5"
    
    def test_with_available(self):
        err = DeviceNotFoundError("gpu:5", available_devices=["gpu:0", "gpu:1"])
        assert "gpu:0" in str(err)
        assert err.available_devices == ["gpu:0", "gpu:1"]


class TestModelLoadError:
    """Test ModelLoadError"""
    
    def test_basic(self):
        err = ModelLoadError("gpt2")
        assert "gpt2" in str(err)
        assert err.model_name == "gpt2"
    
    def test_with_reason(self):
        err = ModelLoadError("gpt2", reason="Out of memory")
        assert "Out of memory" in str(err)


class TestServerStartError:
    """Test ServerStartError"""
    
    def test_basic(self):
        err = ServerStartError("Triton")
        assert "Triton" in str(err)
        assert err.engine_name == "Triton"
    
    def test_with_reason(self):
        err = ServerStartError("Triton", reason="Port already in use")
        assert "Port already in use" in str(err)


class TestInvalidDeviceError:
    """Test InvalidDeviceError"""
    
    def test_basic(self):
        err = InvalidDeviceError("invalid:device")
        assert "invalid:device" in str(err)
        assert "Valid formats" in str(err)
    
    def test_with_valid_formats(self):
        err = InvalidDeviceError("bad", valid_formats=["gpu:0", "cpu"])
        assert "gpu:0" in str(err)


class TestDockerNotAvailableError:
    """Test DockerNotAvailableError"""
    
    def test_default(self):
        err = DockerNotAvailableError()
        assert "Docker" in str(err)
        assert "usermod" in str(err)  # Contains fix hint
    
    def test_with_reason(self):
        err = DockerNotAvailableError(reason="Permission denied")
        assert "Permission denied" in str(err)


class TestPackageNotInstalledError:
    """Test PackageNotInstalledError"""
    
    def test_basic(self):
        err = PackageNotInstalledError("vllm")
        assert "vllm" in str(err)
        assert err.package == "vllm"
    
    def test_with_install_cmd(self):
        err = PackageNotInstalledError("vllm", install_cmd="pip install vllm")
        assert "pip install vllm" in str(err)


class TestExceptionCatching:
    """Test catching exceptions with base classes"""
    
    def test_catch_with_base(self):
        """All exceptions can be caught with RMAbstractError"""
        exceptions_to_test = [
            NotInitializedError(),
            BackendNotAvailableError("test"),
            ModelLoadError("test"),
            ServerStartError("test"),
            InvalidDeviceError("test"),
            DockerNotAvailableError(),
            PackageNotInstalledError("test"),
        ]
        
        for exc in exceptions_to_test:
            with pytest.raises(RMAbstractError):
                raise exc
    
    def test_catch_backend_errors(self):
        """Backend errors can be caught with BackendError"""
        with pytest.raises(BackendError):
            raise BackendNotAvailableError("gpu")
        
        with pytest.raises(BackendError):
            raise DeviceNotFoundError("gpu:5")
    
    def test_catch_model_errors(self):
        """Model errors can be caught with ModelError"""
        with pytest.raises(ModelError):
            raise ModelLoadError("gpt2")
        
        with pytest.raises(ModelError):
            raise ModelNotLoadedError()


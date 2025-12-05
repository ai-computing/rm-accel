"""
RM Abstract Layer - Custom Exceptions

Provides clear, actionable error messages for common failure scenarios.
"""


class RMAbstractError(Exception):
    """Base exception for all RM Abstract Layer errors"""
    pass


# ============================================================
# Initialization Errors
# ============================================================

class InitializationError(RMAbstractError):
    """Error during initialization"""
    pass


class NotInitializedError(RMAbstractError):
    """RM Abstract Layer not initialized"""
    
    def __init__(self, message: str = None):
        super().__init__(
            message or "RM Abstract Layer not initialized. Call rm_abstract.init() first."
        )


# ============================================================
# Backend Errors
# ============================================================

class BackendError(RMAbstractError):
    """Base class for backend-related errors"""
    pass


class BackendNotAvailableError(BackendError):
    """Requested backend is not available"""
    
    def __init__(self, backend_name: str, reason: str = None, install_hint: str = None):
        self.backend_name = backend_name
        self.reason = reason
        self.install_hint = install_hint
        
        message = f"Backend '{backend_name}' is not available"
        if reason:
            message += f": {reason}"
        if install_hint:
            message += f"\nInstall with: {install_hint}"
        
        super().__init__(message)


class BackendInitError(BackendError):
    """Failed to initialize backend"""
    pass


class DeviceNotFoundError(BackendError):
    """Specified device not found"""
    
    def __init__(self, device: str, available_devices: list = None):
        self.device = device
        self.available_devices = available_devices
        
        message = f"Device '{device}' not found"
        if available_devices:
            message += f". Available devices: {available_devices}"
        
        super().__init__(message)


# ============================================================
# Model Errors
# ============================================================

class ModelError(RMAbstractError):
    """Base class for model-related errors"""
    pass


class ModelLoadError(ModelError):
    """Failed to load model"""
    
    def __init__(self, model_name: str, reason: str = None):
        self.model_name = model_name
        self.reason = reason
        
        message = f"Failed to load model '{model_name}'"
        if reason:
            message += f": {reason}"
        
        super().__init__(message)


class ModelNotLoadedError(ModelError):
    """Model not loaded"""
    
    def __init__(self, message: str = None):
        super().__init__(
            message or "Model not loaded. Call load_model() first."
        )


class InferenceError(ModelError):
    """Error during inference"""
    pass


# ============================================================
# Serving Errors
# ============================================================

class ServingError(RMAbstractError):
    """Base class for serving-related errors"""
    pass


class ServerStartError(ServingError):
    """Failed to start server"""
    
    def __init__(self, engine_name: str, reason: str = None):
        self.engine_name = engine_name
        self.reason = reason
        
        message = f"Failed to start {engine_name} server"
        if reason:
            message += f": {reason}"
        
        super().__init__(message)


class ServerNotRunningError(ServingError):
    """Server is not running"""
    
    def __init__(self, engine_name: str = None):
        message = "Server is not running"
        if engine_name:
            message = f"{engine_name} server is not running"
        message += ". Call start_server() first."
        
        super().__init__(message)


class ServerConnectionError(ServingError):
    """Cannot connect to server"""
    
    def __init__(self, url: str, reason: str = None):
        self.url = url
        self.reason = reason
        
        message = f"Cannot connect to server at {url}"
        if reason:
            message += f": {reason}"
        
        super().__init__(message)


# ============================================================
# Configuration Errors
# ============================================================

class ConfigurationError(RMAbstractError):
    """Invalid configuration"""
    pass


class InvalidDeviceError(ConfigurationError):
    """Invalid device specification"""
    
    def __init__(self, device: str, valid_formats: list = None):
        self.device = device
        
        message = f"Invalid device specification: '{device}'"
        if valid_formats:
            message += f". Valid formats: {valid_formats}"
        else:
            message += ". Valid formats: 'auto', 'gpu:0', 'cpu', 'rbln:0'"
        
        super().__init__(message)


# ============================================================
# Docker Errors (for Triton)
# ============================================================

class DockerError(RMAbstractError):
    """Base class for Docker-related errors"""
    pass


class DockerNotAvailableError(DockerError):
    """Docker is not available"""
    
    def __init__(self, reason: str = None):
        message = "Docker is not available"
        if reason:
            message += f": {reason}"
        message += "\nEnsure Docker is installed and running, and you have permission to use it."
        message += "\nTry: sudo usermod -aG docker $USER && newgrp docker"
        
        super().__init__(message)


class DockerImageNotFoundError(DockerError):
    """Docker image not found"""
    
    def __init__(self, image: str, build_hint: str = None):
        self.image = image
        
        message = f"Docker image '{image}' not found"
        if build_hint:
            message += f"\nBuild with: {build_hint}"
        
        super().__init__(message)


# ============================================================
# Dependency Errors
# ============================================================

class DependencyError(RMAbstractError):
    """Missing or incompatible dependency"""
    pass


class PackageNotInstalledError(DependencyError):
    """Required package not installed"""
    
    def __init__(self, package: str, install_cmd: str = None):
        self.package = package
        self.install_cmd = install_cmd
        
        message = f"Required package '{package}' is not installed"
        if install_cmd:
            message += f"\nInstall with: {install_cmd}"
        
        super().__init__(message)


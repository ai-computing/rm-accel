"""
DeviceFlowController - Device Flow Controller

Core controller responsible for device selection and resource management.
Uses the plugin registry system for resource discovery.
"""

from typing import Any, Dict, Optional, Type
import logging

from .config import Config
from ..resources.base import Resource, ResourceInfo
from ..registry import PluginRegistry, PluginType, get_registry, discover_resources

logger = logging.getLogger(__name__)


class DeviceFlowController:
    """
    Device Flow Controller

    - Device string parsing and resource selection
    - Model preparation and execution
    - Inter-resource switching support

    Uses the plugin registry system for resource discovery and selection.
    """

    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object
        """
        self.config = config
        self._resource: Optional[Resource] = None
        self._prepared_models: Dict[int, Any] = {}
        self._registry = get_registry()

        # Auto-discover resources if registry is empty
        if not self._registry.list_plugins(PluginType.RESOURCE):
            discover_resources(self._registry)

        # Initialize resource
        self._initialize_resource()

    @classmethod
    def get_available_backends(cls) -> Dict[str, bool]:
        """
        Return list of available resources.

        Returns:
            {resource_name: availability} dictionary
        """
        registry = get_registry()
        if not registry.list_plugins(PluginType.RESOURCE):
            discover_resources(registry)

        result = {}
        for plugin_info in registry.list_plugins(PluginType.RESOURCE):
            result[plugin_info.name] = plugin_info.is_available()

        return result

    def _initialize_resource(self) -> None:
        """Initialize resource for the device"""
        device_type = self.config.device_type
        device_id = self.config.device_id

        if device_type == "auto":
            self._resource = self._auto_select_resource()
        else:
            self._resource = self._create_resource(device_type, device_id)

        if self._resource is not None:
            self._resource.initialize()

    def _auto_select_resource(self) -> Optional[Resource]:
        """
        Auto-select optimal resource.

        Uses plugin registry priority (NPU > GPU > CPU).
        """
        best_plugin = self._registry.get_best_available(PluginType.RESOURCE)
        if best_plugin:
            try:
                resource = self._create_resource_instance(
                    best_plugin.cls, 0, best_plugin.name
                )
                if resource and resource.is_available():
                    logger.info(f"Auto-selected resource: {best_plugin.name}")
                    return resource
            except Exception as e:
                logger.debug(f"Resource {best_plugin.name} not available: {e}")

        logger.warning("No resource available")
        return None

    def _create_resource(self, device_type: str, device_id: int) -> Optional[Resource]:
        """Create resource instance from plugin registry"""
        device_type_lower = device_type.lower()

        plugin_info = self._registry.get_info(device_type_lower)
        if plugin_info:
            return self._create_resource_instance(
                plugin_info.cls, device_id, plugin_info.name
            )

        raise ValueError(f"Unknown device type: {device_type}")

    def _create_resource_instance(
        self, resource_class: Type, device_id: int, name: str
    ) -> Optional[Resource]:
        """Create a resource instance with appropriate options."""
        # NPU resources need additional options
        if name in ["rbln", "furiosa", "rebellions", "rngd"]:
            resource = resource_class(
                device_id=device_id,
                cache_dir=self.config.cache_dir,
            )
        else:
            resource = resource_class(device_id=device_id)

        return resource

    @property
    def resource(self) -> Optional[Resource]:
        """Return current resource"""
        return self._resource

    @property
    def device_name(self) -> str:
        """Return current device name"""
        if self._resource is None:
            return "none"
        return f"{self._resource.name}:{self._resource.device_id}"

    def switch_device(self, device: str) -> None:
        """
        Switch device at runtime.

        Args:
            device: New device specification
        """
        if self._resource:
            self._resource.cleanup()

        self.config.device = device
        self._prepared_models.clear()
        self._initialize_resource()

        logger.info(f"Switched to device: {self.device_name}")

    def prepare_model(
        self, model: Any, model_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Prepare model (convert/compile for the resource).

        Args:
            model: Original model
            model_config: Model configuration

        Returns:
            Prepared model (compiled/optimized for the resource)
        """
        if self._resource is None:
            logger.warning("No resource available, returning original model")
            return model

        model_id = id(model)

        if model_id in self._prepared_models:
            return self._prepared_models[model_id]

        prepared = self._resource.prepare_model(model, model_config)
        self._prepared_models[model_id] = prepared

        return prepared

    def execute(self, model: Any, inputs: Any, **kwargs) -> Any:
        """
        Execute inference.

        Args:
            model: Prepared model
            inputs: Input data
            **kwargs: Additional options

        Returns:
            Inference result
        """
        if self._resource is None:
            raise RuntimeError("No resource available")

        return self._resource.execute(model, inputs, **kwargs)

    def get_device_info(self) -> Dict[str, Any]:
        """Return current device information"""
        if self._resource is None:
            return {"status": "no_resource"}

        info = self._resource.get_info()
        return {
            "resource_type": info.resource_type.value,
            "device_id": info.device_id,
            "name": info.name,
            "vendor": info.vendor,
            "memory_total": info.memory_total,
            "memory_available": info.memory_available,
            "extra": info.extra,
        }

    def get_available_resources(self) -> Dict[str, bool]:
        """
        Get available resources from plugin registry.

        Returns:
            {resource_name: availability} dictionary
        """
        result = {}
        for plugin_info in self._registry.list_plugins(PluginType.RESOURCE):
            result[plugin_info.name] = plugin_info.is_available()
        return result

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._resource:
            self._resource.cleanup()
        self._prepared_models.clear()

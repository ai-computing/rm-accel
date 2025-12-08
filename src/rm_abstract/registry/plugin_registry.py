"""
Plugin Registry

Centralized registry for all plugins (resources, engines).
Eliminates hardcoded backend/engine names throughout the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type
import logging

logger = logging.getLogger(__name__)


class PluginType(str, Enum):
    """Plugin types"""

    RESOURCE = "resource"  # Hardware resource (GPU, NPU, CPU)
    ENGINE = "engine"  # Serving engine (vLLM, Triton, TorchServe)


@dataclass
class PluginInfo:
    """Plugin metadata"""

    name: str  # Unique plugin name (e.g., "nvidia", "rbln", "vllm")
    plugin_type: PluginType
    cls: Type  # Plugin class
    vendor: str = ""
    description: str = ""
    priority: int = 0  # Higher = preferred for auto-selection
    aliases: List[str] = field(default_factory=list)  # Alternative names
    requirements: List[str] = field(default_factory=list)  # Required packages
    check_available: Optional[Callable[[], bool]] = None  # Custom availability check

    def is_available(self) -> bool:
        """Check if plugin is available"""
        if self.check_available:
            return self.check_available()

        # Use class method if available
        if hasattr(self.cls, "is_available"):
            try:
                return self.cls.is_available()
            except Exception:
                return False

        return True


class PluginRegistry:
    """
    Central registry for all plugins.

    Provides unified interface for plugin registration, discovery,
    and instantiation without hardcoding names throughout the codebase.

    Example:
        # Register a plugin
        registry = PluginRegistry()
        registry.register(PluginInfo(
            name="nvidia",
            plugin_type=PluginType.RESOURCE,
            cls=NvidiaGPU,
            vendor="nvidia",
            priority=10,
            aliases=["gpu", "cuda"],
        ))

        # Get plugin by name
        gpu_class = registry.get("nvidia")
        gpu = gpu_class(device_id=0)

        # Auto-select best available
        best = registry.get_best_available(PluginType.RESOURCE)
    """

    _instance: Optional["PluginRegistry"] = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plugins: Dict[str, PluginInfo] = {}
            cls._instance._aliases: Dict[str, str] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "PluginRegistry":
        """Get singleton instance"""
        return cls()

    def register(self, plugin_info: PluginInfo) -> None:
        """
        Register a plugin.

        Args:
            plugin_info: Plugin metadata
        """
        name = plugin_info.name.lower()

        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already registered, overwriting")

        self._plugins[name] = plugin_info

        # Register aliases
        for alias in plugin_info.aliases:
            alias = alias.lower()
            self._aliases[alias] = name
            logger.debug(f"Registered alias '{alias}' -> '{name}'")

        logger.debug(f"Registered plugin: {name} ({plugin_info.plugin_type.value})")

    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin.

        Args:
            name: Plugin name

        Returns:
            True if plugin was unregistered
        """
        name = name.lower()

        # Resolve alias
        if name in self._aliases:
            name = self._aliases[name]

        if name not in self._plugins:
            return False

        plugin_info = self._plugins.pop(name)

        # Remove aliases
        for alias in plugin_info.aliases:
            self._aliases.pop(alias.lower(), None)

        return True

    def get(self, name: str) -> Optional[Type]:
        """
        Get plugin class by name.

        Args:
            name: Plugin name or alias

        Returns:
            Plugin class or None
        """
        plugin_info = self.get_info(name)
        return plugin_info.cls if plugin_info else None

    def get_info(self, name: str) -> Optional[PluginInfo]:
        """
        Get plugin info by name.

        Args:
            name: Plugin name or alias

        Returns:
            PluginInfo or None
        """
        name = name.lower()

        # Check alias
        if name in self._aliases:
            name = self._aliases[name]

        return self._plugins.get(name)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        available_only: bool = False,
    ) -> List[PluginInfo]:
        """
        List registered plugins.

        Args:
            plugin_type: Filter by type
            available_only: Only return available plugins

        Returns:
            List of PluginInfo
        """
        plugins = list(self._plugins.values())

        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]

        if available_only:
            plugins = [p for p in plugins if p.is_available()]

        return plugins

    def get_available(
        self,
        plugin_type: Optional[PluginType] = None,
    ) -> List[PluginInfo]:
        """
        Get available plugins.

        Args:
            plugin_type: Filter by type

        Returns:
            List of available PluginInfo sorted by priority
        """
        plugins = self.list_plugins(plugin_type, available_only=True)
        return sorted(plugins, key=lambda p: -p.priority)

    def get_best_available(
        self,
        plugin_type: PluginType,
        exclude: Optional[List[str]] = None,
    ) -> Optional[PluginInfo]:
        """
        Get best available plugin based on priority.

        Args:
            plugin_type: Plugin type
            exclude: Plugin names to exclude

        Returns:
            Best available PluginInfo or None
        """
        exclude = [e.lower() for e in (exclude or [])]
        available = self.get_available(plugin_type)

        for plugin in available:
            if plugin.name.lower() not in exclude:
                return plugin

        return None

    def create(self, name: str, **kwargs) -> Any:
        """
        Create plugin instance.

        Args:
            name: Plugin name
            **kwargs: Arguments to pass to plugin constructor

        Returns:
            Plugin instance

        Raises:
            ValueError: If plugin not found
        """
        plugin_class = self.get(name)
        if plugin_class is None:
            raise ValueError(f"Plugin '{name}' not found")

        return plugin_class(**kwargs)

    def is_registered(self, name: str) -> bool:
        """Check if plugin is registered"""
        name = name.lower()
        return name in self._plugins or name in self._aliases

    def is_available(self, name: str) -> bool:
        """Check if plugin is available"""
        plugin_info = self.get_info(name)
        return plugin_info.is_available() if plugin_info else False

    def clear(self) -> None:
        """Clear all registered plugins"""
        self._plugins.clear()
        self._aliases.clear()

    def summary(self) -> str:
        """Get registry summary"""
        lines = ["Plugin Registry:"]

        for plugin_type in PluginType:
            plugins = self.list_plugins(plugin_type)
            if plugins:
                lines.append(f"\n  {plugin_type.value}s:")
                for p in sorted(plugins, key=lambda x: -x.priority):
                    status = "available" if p.is_available() else "not available"
                    lines.append(f"    - {p.name} ({p.vendor}) [{status}]")

        return "\n".join(lines)


# Global registry instance
_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get global plugin registry"""
    return _registry


def register_plugin(plugin_info: PluginInfo) -> None:
    """Register a plugin to global registry"""
    _registry.register(plugin_info)


def get_plugin(name: str) -> Optional[Type]:
    """Get plugin class from global registry"""
    return _registry.get(name)

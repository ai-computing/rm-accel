"""
Plugin Registry System

Centralized plugin registration and discovery.
Eliminates hardcoded backend names scattered throughout the codebase.

Structure:
- plugin_registry.py: Central registry for all plugins
- discovery.py: Auto-discovery of available plugins
"""

from .discovery import discover_all, discover_engines, discover_resources
from .plugin_registry import (
    PluginInfo,
    PluginRegistry,
    PluginType,
    get_plugin,
    get_registry,
    register_plugin,
)

__all__ = [
    "PluginRegistry",
    "PluginInfo",
    "PluginType",
    "get_registry",
    "register_plugin",
    "get_plugin",
    "discover_resources",
    "discover_engines",
    "discover_all",
]

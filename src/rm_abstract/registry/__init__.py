"""
Plugin Registry System

Centralized plugin registration and discovery.
Eliminates hardcoded backend names scattered throughout the codebase.

Structure:
- plugin_registry.py: Central registry for all plugins
- discovery.py: Auto-discovery of available plugins
"""

from .plugin_registry import (
    PluginRegistry,
    PluginInfo,
    PluginType,
    get_registry,
    register_plugin,
    get_plugin,
)
from .discovery import discover_resources, discover_engines, discover_all

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

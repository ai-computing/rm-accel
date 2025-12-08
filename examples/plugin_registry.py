"""
Plugin Registry Example

Demonstrates how to use the plugin registry system.
"""

from rm_abstract import get_registry, PluginType
from rm_abstract.registry import discover_all, PluginInfo


def main():
    # Get the global registry
    registry = get_registry()

    # Discover all plugins
    print("=" * 50)
    print("Discovering Plugins")
    print("=" * 50)

    discovered = discover_all(registry)
    print(f"Discovered {len(discovered)} plugins")

    # List all resources
    print("\n" + "=" * 50)
    print("Available Resources")
    print("=" * 50)

    resources = registry.list_plugins(PluginType.RESOURCE)
    for plugin in sorted(resources, key=lambda p: -p.priority):
        status = "available" if plugin.is_available() else "not available"
        print(f"  {plugin.name} ({plugin.vendor}) - priority {plugin.priority} [{status}]")

    # List all engines
    print("\n" + "=" * 50)
    print("Available Engines")
    print("=" * 50)

    engines = registry.list_plugins(PluginType.ENGINE)
    for plugin in sorted(engines, key=lambda p: -p.priority):
        status = "available" if plugin.is_available() else "not available"
        print(f"  {plugin.name} ({plugin.vendor}) - priority {plugin.priority} [{status}]")

    # Get best available resource
    print("\n" + "=" * 50)
    print("Best Available Resource")
    print("=" * 50)

    best = registry.get_best_available(PluginType.RESOURCE)
    if best:
        print(f"  Name: {best.name}")
        print(f"  Vendor: {best.vendor}")
        print(f"  Description: {best.description}")
        print(f"  Priority: {best.priority}")
    else:
        print("  No resource available")


if __name__ == "__main__":
    main()

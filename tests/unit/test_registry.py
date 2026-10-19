"""Plugin registry tests"""

from rm_abstract.registry import (
    PluginInfo,
    PluginType,
    get_registry,
)
from rm_abstract.resources.base import (
    Resource,
    ResourceCapability,
    ResourceInfo,
    ResourceType,
)


class MockResource(Resource):
    """Mock resource for testing"""

    RESOURCE_TYPE = ResourceType.CPU
    VENDOR = "mock"
    NAME = "MockResource"

    @staticmethod
    def is_available() -> bool:
        return True

    def initialize(self) -> None:
        self._initialized = True

    def prepare_model(self, model, config=None):
        return model

    def execute(self, model, inputs, **kwargs):
        return {"output": "mock_result"}

    def get_info(self) -> ResourceInfo:
        return ResourceInfo(
            resource_type=ResourceType.CPU,
            device_id=0,
            name="MockResource",
        )

    def get_capability(self) -> ResourceCapability:
        return ResourceCapability()


class TestPluginRegistry:
    """PluginRegistry tests"""

    def setup_method(self):
        """Clear registry before each test"""
        registry = get_registry()
        registry.clear()

    def test_register_plugin(self):
        """Test plugin registration"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
            vendor="mock",
            priority=10,
        )
        registry.register(plugin_info)
        assert registry.is_registered("mock")

    def test_get_plugin(self):
        """Test getting plugin by name"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        registry.register(plugin_info)

        cls = registry.get("mock")
        assert cls is MockResource

    def test_get_plugin_by_alias(self):
        """Test getting plugin by alias"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
            aliases=["mock_alias", "another_alias"],
        )
        registry.register(plugin_info)

        assert registry.get("mock_alias") is MockResource
        assert registry.get("another_alias") is MockResource

    def test_list_plugins(self):
        """Test listing plugins"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        registry.register(plugin_info)

        plugins = registry.list_plugins(PluginType.RESOURCE)
        assert len(plugins) == 1
        assert plugins[0].name == "mock"

    def test_get_available(self):
        """Test getting available plugins"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        registry.register(plugin_info)

        available = registry.get_available(PluginType.RESOURCE)
        assert len(available) == 1

    def test_get_best_available(self):
        """Test getting best available plugin by priority"""
        registry = get_registry()

        plugin1 = PluginInfo(
            name="low_priority",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
            priority=10,
        )
        plugin2 = PluginInfo(
            name="high_priority",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
            priority=100,
        )

        registry.register(plugin1)
        registry.register(plugin2)

        best = registry.get_best_available(PluginType.RESOURCE)
        assert best.name == "high_priority"

    def test_unregister_plugin(self):
        """Test unregistering plugin"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        registry.register(plugin_info)
        assert registry.is_registered("mock")

        registry.unregister("mock")
        assert not registry.is_registered("mock")

    def test_create_instance(self):
        """Test creating plugin instance"""
        registry = get_registry()
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        registry.register(plugin_info)

        instance = registry.create("mock", device_id=1)
        assert instance.device_id == 1


class TestPluginInfo:
    """PluginInfo tests"""

    def test_is_available_static_method(self):
        """Test is_available uses static method"""
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
        )
        assert plugin_info.is_available() is True

    def test_is_available_custom_check(self):
        """Test is_available with custom check function"""
        plugin_info = PluginInfo(
            name="mock",
            plugin_type=PluginType.RESOURCE,
            cls=MockResource,
            check_available=lambda: False,
        )
        assert plugin_info.is_available() is False

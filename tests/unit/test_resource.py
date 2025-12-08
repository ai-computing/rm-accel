"""Resource base class tests"""

import pytest
from rm_abstract.resources.base import Resource, ResourceType, ResourceInfo, ResourceCapability


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
            vendor="mock",
        )

    def get_capability(self) -> ResourceCapability:
        return ResourceCapability()


class TestResource:
    """Resource ABC tests"""

    def test_mock_resource_creation(self):
        """Test mock resource creation"""
        resource = MockResource(device_id=0)
        assert resource.device_id == 0
        assert resource.name == "MockResource"

    def test_resource_is_available(self):
        """Test is_available method"""
        assert MockResource.is_available() is True

    def test_resource_initialize(self):
        """Test initialize method"""
        resource = MockResource()
        assert resource._initialized is False
        resource.initialize()
        assert resource._initialized is True

    def test_resource_prepare_model(self):
        """Test prepare_model method"""
        resource = MockResource()
        model = {"type": "test_model"}
        prepared = resource.prepare_model(model)
        assert prepared == model

    def test_resource_execute(self):
        """Test execute method"""
        resource = MockResource()
        result = resource.execute(None, None)
        assert result == {"output": "mock_result"}

    def test_resource_get_info(self):
        """Test get_info method"""
        resource = MockResource()
        info = resource.get_info()
        assert info.resource_type == ResourceType.CPU
        assert info.name == "MockResource"

    def test_resource_cleanup(self):
        """Test cleanup method"""
        resource = MockResource()
        resource.initialize()
        assert resource._initialized is True
        resource.cleanup()
        assert resource._initialized is False


class TestResourceInfo:
    """ResourceInfo dataclass tests"""

    def test_resource_info_creation(self):
        """Test ResourceInfo creation"""
        info = ResourceInfo(
            resource_type=ResourceType.GPU,
            device_id=0,
            name="NVIDIA A100",
            vendor="NVIDIA",
            memory_total=80 * 1024**3,
        )
        assert info.resource_type == ResourceType.GPU
        assert info.device_id == 0
        assert info.name == "NVIDIA A100"
        assert info.vendor == "NVIDIA"

    def test_resource_info_optional_fields(self):
        """Test ResourceInfo optional fields"""
        info = ResourceInfo(
            resource_type=ResourceType.NPU,
            device_id=0,
            name="ATOM",
            vendor="rebellions",
        )
        assert info.vendor == "rebellions"
        assert info.memory_total is None
        assert info.extra == {}


class TestResourceType:
    """ResourceType Enum tests"""

    def test_resource_types(self):
        """Test resource type values"""
        assert ResourceType.GPU.value == "gpu"
        assert ResourceType.NPU.value == "npu"
        assert ResourceType.CPU.value == "cpu"

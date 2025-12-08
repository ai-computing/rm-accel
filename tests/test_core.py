"""
Core functionality tests
"""

import pytest


class TestInitialization:
    """Test rm_abstract initialization"""

    def test_import(self):
        """Test that rm_abstract can be imported"""
        import rm_abstract

        assert rm_abstract is not None

    def test_version(self):
        """Test version is defined"""
        import rm_abstract

        assert hasattr(rm_abstract, "__version__")
        assert rm_abstract.__version__ is not None

    def test_init_auto(self):
        """Test auto device initialization"""
        import rm_abstract

        controller = rm_abstract.init(device="auto", verbose=False)
        assert controller is not None

    def test_init_cpu(self):
        """Test CPU device initialization"""
        import rm_abstract

        controller = rm_abstract.init(device="cpu", verbose=False)
        assert controller is not None
        assert "cpu" in controller.device_name.lower()

    def test_get_controller(self):
        """Test get_controller returns initialized controller"""
        import rm_abstract

        rm_abstract.init(device="cpu", verbose=False)
        controller = rm_abstract.get_controller()
        assert controller is not None


class TestDeviceSwitching:
    """Test device switching functionality"""

    def test_switch_to_cpu(self):
        """Test switching to CPU"""
        import rm_abstract

        rm_abstract.init(device="auto", verbose=False)
        rm_abstract.switch_device("cpu")

        controller = rm_abstract.get_controller()
        assert "cpu" in controller.device_name.lower()

    def test_get_device_info(self):
        """Test get_device_info returns valid info"""
        import rm_abstract

        rm_abstract.init(device="cpu", verbose=False)
        info = rm_abstract.get_device_info()

        assert isinstance(info, dict)
        assert "resource_type" in info
        assert "device_id" in info


class TestResources:
    """Test resource availability"""

    def test_get_available_resources(self):
        """Test getting available resources"""
        import rm_abstract

        resources = rm_abstract.get_available_resources()

        assert isinstance(resources, dict)
        assert "cpu" in resources
        # CPU should always be available if PyTorch is installed
        assert resources["cpu"] is True

    def test_nvidia_resource_detection(self):
        """Test NVIDIA resource detection"""
        import rm_abstract

        resources = rm_abstract.get_available_resources()

        # Check nvidia resource exists (may or may not be available)
        assert "nvidia" in resources
        assert isinstance(resources["nvidia"], bool)

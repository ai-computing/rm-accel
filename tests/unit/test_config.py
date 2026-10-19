"""Config 클래스 테스트"""

from rm_abstract.core.config import Config


def test_config_default_values():
    """기본 설정 값 테스트"""
    config = Config()

    assert config.device == "auto"
    assert config.verbose is True
    assert config.cache_dir is not None


def test_config_device_parsing():
    """디바이스 파싱 테스트"""
    # GPU 파싱
    config = Config(device="gpu:0")
    assert config.device_type == "gpu"
    assert config.device_id == 0

    # NPU 파싱
    config = Config(device="rbln:1")
    assert config.device_type == "rbln"
    assert config.device_id == 1

    # Auto
    config = Config(device="auto")
    assert config.device_type == "auto"
    assert config.device_id == 0


def test_config_to_dict():
    """설정 딕셔너리 변환 테스트"""
    config = Config(device="gpu:0", verbose=False)
    data = config.to_dict()

    assert data["device"] == "gpu:0"
    assert data["verbose"] is False


def test_config_from_dict():
    """딕셔너리에서 설정 생성 테스트"""
    data = {"device": "furiosa:0", "verbose": True}
    config = Config.from_dict(data)

    assert config.device == "furiosa:0"
    assert config.verbose is True

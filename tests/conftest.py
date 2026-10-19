"""
Pytest configuration and fixtures
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_model_name():
    """Default test model (small for fast tests)"""
    return "gpt2"


@pytest.fixture(scope="session")
def test_prompt():
    """Default test prompt"""
    return "Hello, I am"


@pytest.fixture
def rm_abstract_initialized():
    """Initialize rm_abstract for testing"""
    import rm_abstract

    try:
        rm_abstract.init(device="auto", verbose=False)
        return True
    except Exception:
        return False

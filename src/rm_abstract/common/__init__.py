"""
Common utilities shared across all backends and engines.

This module provides:
- InputAdapter: Unified input format handling
- OutputAdapter: Unified output format conversion
- ModelConverter: ONNX and other format conversions
- HardwareDetector: Hardware detection utilities
"""

from .input_adapter import InputAdapter, InputFormat
from .output_adapter import OutputAdapter, OutputFormat
from .model_converter import ModelConverter, ConversionConfig
from .hardware_detector import HardwareDetector, HardwareInfo

__all__ = [
    "InputAdapter",
    "InputFormat",
    "OutputAdapter",
    "OutputFormat",
    "ModelConverter",
    "ConversionConfig",
    "HardwareDetector",
    "HardwareInfo",
]

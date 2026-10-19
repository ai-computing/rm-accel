"""
Common utilities shared across all backends and engines.

This module provides:
- InputAdapter: Unified input format handling
- OutputAdapter: Unified output format conversion
- ModelConverter: ONNX and other format conversions
- HardwareDetector: Hardware detection utilities
"""

from .hardware_detector import HardwareDetector, HardwareInfo
from .input_adapter import InputAdapter, InputFormat
from .model_converter import ConversionConfig, ModelConverter
from .output_adapter import OutputAdapter, OutputFormat

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

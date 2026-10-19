"""
Hardware Resources

Each hardware type is clearly separated into its own module.
All resources implement the common Resource interface.

Structure:
- base.py: Base Resource class and interfaces
- gpu/: GPU resources (NVIDIA, AMD)
- npu/: NPU resources (Rebellions, FuriosaAI)
- cpu/: CPU fallback resource
"""

from .base import Resource, ResourceCapability, ResourceInfo

__all__ = [
    "Resource",
    "ResourceInfo",
    "ResourceCapability",
]

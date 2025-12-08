"""
NPU Resources

Contains NPU-specific resource implementations.
Each vendor has its own file for clear separation.
"""

from .rebellions import RebellionsATOM
from .furiosa import FuriosaRNGD

__all__ = ["RebellionsATOM", "FuriosaRNGD"]

"""
NPU Resources

Contains NPU-specific resource implementations.
Each vendor has its own file for clear separation.
"""

from .furiosa import FuriosaRNGD
from .rebellions import RebellionsATOM

__all__ = ["RebellionsATOM", "FuriosaRNGD"]

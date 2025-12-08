"""
Compatibility layer for transparent backend switching.

This module provides drop-in replacements for popular ML frameworks
that automatically route to the appropriate backend based on device selection.
"""

from .vllm_compat import LLM, SamplingParams, RequestOutput
from .transformers_compat import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "AutoModelForCausalLM",
    "AutoTokenizer",
]

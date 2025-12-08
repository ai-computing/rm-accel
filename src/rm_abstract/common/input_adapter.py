"""
Unified Input Adapter

Handles conversion between various input formats used by different backends.
Eliminates duplicated input handling logic across GPU, NPU, and CPU backends.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    """Input format types"""

    STRING = "string"  # Raw text string
    STRING_LIST = "string_list"  # List of text strings
    TOKEN_IDS = "token_ids"  # Token ID tensor
    TOKEN_IDS_DICT = "token_ids_dict"  # Dict with input_ids key
    HF_BATCH_ENCODING = "hf_batch_encoding"  # HuggingFace BatchEncoding
    NUMPY = "numpy"  # NumPy array
    UNKNOWN = "unknown"


@dataclass
class AdaptedInput:
    """Standardized input container"""

    prompts: Optional[List[str]] = None  # Text prompts (for vLLM-style)
    token_ids: Optional[Any] = None  # Token IDs tensor
    attention_mask: Optional[Any] = None  # Attention mask
    original_format: InputFormat = InputFormat.UNKNOWN
    metadata: Optional[Dict[str, Any]] = None


class InputAdapter:
    """
    Unified input adapter for all backends.

    Converts various input formats to a standardized format that can be
    used by any backend (GPU/NPU/CPU).

    Example:
        adapter = InputAdapter()
        adapted = adapter.adapt(inputs, tokenizer=tokenizer)

        # For vLLM-style backends (need text prompts)
        prompts = adapted.prompts

        # For PyTorch-style backends (need token IDs)
        token_ids = adapted.token_ids
    """

    def __init__(self, tokenizer: Optional[Any] = None):
        """
        Args:
            tokenizer: Optional tokenizer for encoding/decoding
        """
        self._tokenizer = tokenizer

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set or update tokenizer"""
        self._tokenizer = tokenizer

    def detect_format(self, inputs: Any) -> InputFormat:
        """
        Detect the format of input data.

        Args:
            inputs: Input data in any format

        Returns:
            Detected InputFormat
        """
        if inputs is None:
            return InputFormat.UNKNOWN

        # String
        if isinstance(inputs, str):
            return InputFormat.STRING

        # List of strings
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str):
            return InputFormat.STRING_LIST

        # HuggingFace BatchEncoding
        if hasattr(inputs, "input_ids") and hasattr(inputs, "attention_mask"):
            return InputFormat.HF_BATCH_ENCODING

        # Dict with input_ids
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return InputFormat.TOKEN_IDS_DICT

        # NumPy array
        try:
            import numpy as np

            if isinstance(inputs, np.ndarray):
                return InputFormat.NUMPY
        except ImportError:
            pass

        # PyTorch tensor or list of token IDs
        if hasattr(inputs, "shape") or (
            isinstance(inputs, list)
            and len(inputs) > 0
            and isinstance(inputs[0], (int, list))
        ):
            return InputFormat.TOKEN_IDS

        return InputFormat.UNKNOWN

    def adapt(
        self,
        inputs: Any,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ) -> AdaptedInput:
        """
        Adapt inputs to standardized format.

        Args:
            inputs: Input data in any format
            tokenizer: Optional tokenizer (overrides instance tokenizer)
            **kwargs: May contain input_ids, attention_mask, etc.

        Returns:
            AdaptedInput with both prompts and token_ids if possible
        """
        tokenizer = tokenizer or self._tokenizer

        # Handle None inputs - check kwargs
        if inputs is None:
            inputs = kwargs.get("input_ids", kwargs.get("inputs"))
            if inputs is None:
                raise ValueError("No inputs provided")

        input_format = self.detect_format(inputs)

        if input_format == InputFormat.STRING:
            return self._adapt_string(inputs, tokenizer)

        elif input_format == InputFormat.STRING_LIST:
            return self._adapt_string_list(inputs, tokenizer)

        elif input_format == InputFormat.HF_BATCH_ENCODING:
            return self._adapt_hf_encoding(inputs, tokenizer)

        elif input_format == InputFormat.TOKEN_IDS_DICT:
            return self._adapt_token_dict(inputs, tokenizer)

        elif input_format == InputFormat.TOKEN_IDS:
            return self._adapt_token_ids(inputs, tokenizer)

        elif input_format == InputFormat.NUMPY:
            return self._adapt_numpy(inputs, tokenizer)

        else:
            # Fallback: try to use as-is
            return AdaptedInput(
                prompts=[str(inputs)],
                token_ids=inputs,
                original_format=input_format,
            )

    def _adapt_string(self, text: str, tokenizer: Optional[Any]) -> AdaptedInput:
        """Adapt string input"""
        token_ids = None
        attention_mask = None

        if tokenizer is not None:
            try:
                encoded = tokenizer(text, return_tensors="pt")
                token_ids = encoded.input_ids
                attention_mask = encoded.get("attention_mask")
            except Exception as e:
                logger.debug(f"Failed to tokenize string: {e}")

        return AdaptedInput(
            prompts=[text],
            token_ids=token_ids,
            attention_mask=attention_mask,
            original_format=InputFormat.STRING,
        )

    def _adapt_string_list(
        self, texts: List[str], tokenizer: Optional[Any]
    ) -> AdaptedInput:
        """Adapt list of strings"""
        token_ids = None
        attention_mask = None

        if tokenizer is not None:
            try:
                encoded = tokenizer(texts, return_tensors="pt", padding=True)
                token_ids = encoded.input_ids
                attention_mask = encoded.get("attention_mask")
            except Exception as e:
                logger.debug(f"Failed to tokenize strings: {e}")

        return AdaptedInput(
            prompts=texts,
            token_ids=token_ids,
            attention_mask=attention_mask,
            original_format=InputFormat.STRING_LIST,
        )

    def _adapt_hf_encoding(
        self, encoding: Any, tokenizer: Optional[Any]
    ) -> AdaptedInput:
        """Adapt HuggingFace BatchEncoding"""
        token_ids = encoding.input_ids
        attention_mask = getattr(encoding, "attention_mask", None)

        # Try to decode to text prompts
        prompts = self._decode_token_ids(token_ids, tokenizer)

        return AdaptedInput(
            prompts=prompts,
            token_ids=token_ids,
            attention_mask=attention_mask,
            original_format=InputFormat.HF_BATCH_ENCODING,
        )

    def _adapt_token_dict(
        self, token_dict: Dict, tokenizer: Optional[Any]
    ) -> AdaptedInput:
        """Adapt dict with input_ids"""
        token_ids = token_dict["input_ids"]
        attention_mask = token_dict.get("attention_mask")

        # Try to decode to text prompts
        prompts = self._decode_token_ids(token_ids, tokenizer)

        return AdaptedInput(
            prompts=prompts,
            token_ids=token_ids,
            attention_mask=attention_mask,
            original_format=InputFormat.TOKEN_IDS_DICT,
        )

    def _adapt_token_ids(self, token_ids: Any, tokenizer: Optional[Any]) -> AdaptedInput:
        """Adapt token ID tensor or list"""
        # Try to decode to text prompts
        prompts = self._decode_token_ids(token_ids, tokenizer)

        return AdaptedInput(
            prompts=prompts,
            token_ids=token_ids,
            original_format=InputFormat.TOKEN_IDS,
        )

    def _adapt_numpy(self, arr: Any, tokenizer: Optional[Any]) -> AdaptedInput:
        """Adapt NumPy array"""
        import torch

        token_ids = torch.from_numpy(arr)
        prompts = self._decode_token_ids(token_ids, tokenizer)

        return AdaptedInput(
            prompts=prompts,
            token_ids=token_ids,
            original_format=InputFormat.NUMPY,
        )

    def _decode_token_ids(
        self, token_ids: Any, tokenizer: Optional[Any]
    ) -> Optional[List[str]]:
        """Decode token IDs to text strings"""
        if tokenizer is None:
            return None

        try:
            import torch

            # Convert to list if tensor
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()

            # Handle batch vs single sequence
            if isinstance(token_ids[0], list):
                # Batch of sequences
                return [
                    tokenizer.decode(ids, skip_special_tokens=False)
                    for ids in token_ids
                ]
            else:
                # Single sequence
                return [tokenizer.decode(token_ids, skip_special_tokens=False)]

        except Exception as e:
            logger.debug(f"Failed to decode token IDs: {e}")
            return None

    def get_tokenizer_from_model(self, model: Any) -> Optional[Any]:
        """
        Try to extract tokenizer from a model object.

        Args:
            model: Model object (HuggingFace model, etc.)

        Returns:
            Tokenizer if found, None otherwise
        """
        # Direct tokenizer attribute
        if hasattr(model, "tokenizer"):
            return model.tokenizer

        # Get from model config
        if hasattr(model, "config"):
            model_name = getattr(model.config, "name_or_path", None)
            if model_name:
                try:
                    from transformers import AutoTokenizer

                    return AutoTokenizer.from_pretrained(model_name)
                except Exception:
                    pass

        return None

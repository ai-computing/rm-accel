"""
Unified Output Adapter

Handles conversion of inference outputs to various formats.
Eliminates duplicated output handling logic across backends.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Output format types"""

    VLLM = "vllm"  # vLLM RequestOutput format
    HF_TOKENS = "hf_tokens"  # HuggingFace token IDs (torch.Tensor)
    HF_MODEL_OUTPUT = "hf_model_output"  # HuggingFace ModelOutput
    RAW_LOGITS = "raw_logits"  # Raw logits tensor
    TEXT = "text"  # Plain text string
    TEXT_LIST = "text_list"  # List of text strings
    NUMPY = "numpy"  # NumPy array


@dataclass
class AdaptedOutput:
    """Standardized output container"""

    text: Optional[List[str]] = None  # Generated text
    token_ids: Optional[Any] = None  # Token IDs
    logits: Optional[Any] = None  # Raw logits if available
    original_output: Optional[Any] = None  # Original output preserved
    original_format: OutputFormat = OutputFormat.TEXT


class OutputAdapter:
    """
    Unified output adapter for all backends.

    Converts various output formats to a standardized format or to
    a specific target format (e.g., HuggingFace-compatible).

    Example:
        adapter = OutputAdapter()

        # Convert vLLM output to HuggingFace format
        hf_output = adapter.to_hf_format(vllm_output, tokenizer=tokenizer)

        # Get standardized output
        adapted = adapter.adapt(raw_output)
        print(adapted.text)  # Generated text
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

    def detect_format(self, outputs: Any) -> OutputFormat:
        """
        Detect the format of output data.

        Args:
            outputs: Output data in any format

        Returns:
            Detected OutputFormat
        """
        if outputs is None:
            return OutputFormat.TEXT

        # vLLM RequestOutput (has prompt_token_ids and outputs attributes)
        if hasattr(outputs, "prompt_token_ids") and hasattr(outputs, "outputs"):
            return OutputFormat.VLLM

        # List of vLLM outputs
        if isinstance(outputs, list) and len(outputs) > 0:
            first = outputs[0]
            if hasattr(first, "prompt_token_ids") and hasattr(first, "outputs"):
                return OutputFormat.VLLM

        # HuggingFace ModelOutput
        if hasattr(outputs, "logits"):
            return OutputFormat.HF_MODEL_OUTPUT

        # PyTorch tensor
        if hasattr(outputs, "shape"):
            # Check if it's logits (3D) or token IDs (2D or 1D)
            if len(outputs.shape) >= 3:
                return OutputFormat.RAW_LOGITS
            else:
                return OutputFormat.HF_TOKENS

        # NumPy array
        try:
            import numpy as np

            if isinstance(outputs, np.ndarray):
                return OutputFormat.NUMPY
        except ImportError:
            pass

        # String
        if isinstance(outputs, str):
            return OutputFormat.TEXT

        # List of strings
        if isinstance(outputs, list) and all(isinstance(x, str) for x in outputs):
            return OutputFormat.TEXT_LIST

        return OutputFormat.TEXT

    def adapt(self, outputs: Any, tokenizer: Optional[Any] = None) -> AdaptedOutput:
        """
        Adapt outputs to standardized format.

        Args:
            outputs: Output data in any format
            tokenizer: Optional tokenizer for decoding

        Returns:
            AdaptedOutput with text, token_ids, etc.
        """
        tokenizer = tokenizer or self._tokenizer
        output_format = self.detect_format(outputs)

        if output_format == OutputFormat.VLLM:
            return self._adapt_vllm(outputs, tokenizer)

        elif output_format == OutputFormat.HF_TOKENS:
            return self._adapt_hf_tokens(outputs, tokenizer)

        elif output_format == OutputFormat.HF_MODEL_OUTPUT:
            return self._adapt_hf_output(outputs, tokenizer)

        elif output_format == OutputFormat.RAW_LOGITS:
            return self._adapt_logits(outputs, tokenizer)

        elif output_format == OutputFormat.TEXT:
            return AdaptedOutput(
                text=[outputs] if isinstance(outputs, str) else None,
                original_output=outputs,
                original_format=output_format,
            )

        elif output_format == OutputFormat.TEXT_LIST:
            return AdaptedOutput(
                text=outputs,
                original_output=outputs,
                original_format=output_format,
            )

        else:
            return AdaptedOutput(
                original_output=outputs,
                original_format=output_format,
            )

    def _adapt_vllm(self, outputs: Any, tokenizer: Optional[Any]) -> AdaptedOutput:
        """Adapt vLLM RequestOutput format"""
        # Handle single output or list
        if not isinstance(outputs, list):
            outputs = [outputs]

        texts = []
        all_token_ids = []

        for output in outputs:
            prompt_token_ids = output.prompt_token_ids
            for completion in output.outputs:
                # Generated text
                texts.append(completion.text)
                # Full token IDs (prompt + generated)
                full_ids = list(prompt_token_ids) + list(completion.token_ids)
                all_token_ids.append(full_ids)

        # Convert to tensor
        token_ids_tensor = None
        try:
            import torch

            if len(all_token_ids) == 1:
                token_ids_tensor = torch.tensor([all_token_ids[0]])
            else:
                # Pad sequences
                max_len = max(len(ids) for ids in all_token_ids)
                pad_id = 0
                if tokenizer:
                    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
                padded = [
                    ids + [pad_id] * (max_len - len(ids)) for ids in all_token_ids
                ]
                token_ids_tensor = torch.tensor(padded)
        except Exception as e:
            logger.debug(f"Failed to convert token IDs to tensor: {e}")

        return AdaptedOutput(
            text=texts,
            token_ids=token_ids_tensor,
            original_output=outputs,
            original_format=OutputFormat.VLLM,
        )

    def _adapt_hf_tokens(
        self, token_ids: Any, tokenizer: Optional[Any]
    ) -> AdaptedOutput:
        """Adapt HuggingFace token ID tensor"""
        texts = None

        if tokenizer is not None:
            try:
                texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logger.debug(f"Failed to decode token IDs: {e}")

        return AdaptedOutput(
            text=texts,
            token_ids=token_ids,
            original_output=token_ids,
            original_format=OutputFormat.HF_TOKENS,
        )

    def _adapt_hf_output(
        self, model_output: Any, tokenizer: Optional[Any]
    ) -> AdaptedOutput:
        """Adapt HuggingFace ModelOutput"""
        logits = model_output.logits if hasattr(model_output, "logits") else None

        return AdaptedOutput(
            logits=logits,
            original_output=model_output,
            original_format=OutputFormat.HF_MODEL_OUTPUT,
        )

    def _adapt_logits(self, logits: Any, tokenizer: Optional[Any]) -> AdaptedOutput:
        """Adapt raw logits tensor"""
        return AdaptedOutput(
            logits=logits,
            original_output=logits,
            original_format=OutputFormat.RAW_LOGITS,
        )

    def to_hf_format(
        self,
        outputs: Any,
        original_inputs: Any,
        tokenizer: Optional[Any] = None,
    ) -> Any:
        """
        Convert outputs to HuggingFace-compatible format.

        This is useful when code expects model.generate() to return
        token IDs (torch.Tensor), but the backend returns a different format.

        Args:
            outputs: Output data in any format
            original_inputs: Original inputs (to detect expected format)
            tokenizer: Optional tokenizer

        Returns:
            Token IDs tensor if input was tokens, otherwise original outputs
        """
        tokenizer = tokenizer or self._tokenizer

        # If input was string, return outputs as-is
        if isinstance(original_inputs, str):
            return outputs

        # Adapt and get token IDs
        adapted = self.adapt(outputs, tokenizer)

        if adapted.token_ids is not None:
            return adapted.token_ids

        return outputs

    def to_text(self, outputs: Any, tokenizer: Optional[Any] = None) -> List[str]:
        """
        Convert outputs to text strings.

        Args:
            outputs: Output data in any format
            tokenizer: Optional tokenizer

        Returns:
            List of text strings
        """
        adapted = self.adapt(outputs, tokenizer or self._tokenizer)

        if adapted.text:
            return adapted.text

        # Fallback
        return [str(outputs)]

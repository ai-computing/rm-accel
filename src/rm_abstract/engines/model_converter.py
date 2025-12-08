"""
Model Converter

Converts HuggingFace models to formats required by serving engines.
- ONNX for Triton
- MAR (Model Archive) for TorchServe
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ModelConverter:
    """
    Converts HuggingFace models to various serving formats.

    Example:
        converter = ModelConverter(cache_dir="~/.rm_abstract/models")

        # Convert to ONNX for Triton
        onnx_path = converter.to_onnx("gpt2")

        # Convert to MAR for TorchServe
        mar_path = converter.to_mar("gpt2")
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize converter.

        Args:
            cache_dir: Directory to cache converted models
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.rm_abstract/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def to_onnx(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        opset_version: int = 14,
        device: str = "cpu",
    ) -> str:
        """
        Convert HuggingFace model to ONNX format.

        Args:
            model_name: HuggingFace model name or path
            output_dir: Output directory (default: cache_dir/onnx/model_name)
            opset_version: ONNX opset version
            device: Device for conversion (cpu or cuda)

        Returns:
            Path to ONNX model directory
        """
        safe_name = model_name.replace("/", "_")
        output_path = Path(output_dir or self.cache_dir / "onnx" / safe_name)

        # Check if already converted
        onnx_file = output_path / "model.onnx"
        if onnx_file.exists():
            print(f"[ONNX] Using cached model: {onnx_file}", flush=True)
            logger.info(f"Using cached ONNX model: {onnx_file}")
            return str(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"[ONNX] Converting {model_name} to ONNX format...", flush=True)
            logger.info(f"Converting {model_name} to ONNX...")

            # Load model and tokenizer
            print(f"[ONNX] Loading model and tokenizer...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()

            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Prepare dummy input
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dummy_input = tokenizer(
                "Hello, world!",
                return_tensors="pt",
                padding=True,
                max_length=32,
            )

            if device == "cuda" and torch.cuda.is_available():
                dummy_input = {k: v.cuda() for k, v in dummy_input.items()}

            # Export to ONNX
            print(f"[ONNX] Exporting to ONNX (this may take a while)...", flush=True)
            torch.onnx.export(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(onnx_file),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )

            # Save tokenizer
            tokenizer.save_pretrained(str(output_path))

            print(f"[ONNX] Conversion complete: {output_path}", flush=True)
            logger.info(f"ONNX model saved to: {output_path}")
            return str(output_path)

        except ImportError as e:
            raise ImportError(f"Missing dependency for ONNX conversion: {e}")
        except Exception as e:
            # Clean up on failure
            if output_path.exists():
                shutil.rmtree(output_path)
            raise RuntimeError(f"Failed to convert to ONNX: {e}")

    def to_mar(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        handler: str = "text_generation",
    ) -> str:
        """
        Convert HuggingFace model to TorchServe MAR format.

        Args:
            model_name: HuggingFace model name or path
            output_dir: Output directory (default: cache_dir/mar)
            handler: TorchServe handler type

        Returns:
            Path to MAR file
        """
        safe_name = model_name.replace("/", "_")
        output_path = Path(output_dir or self.cache_dir / "mar")
        output_path.mkdir(parents=True, exist_ok=True)

        mar_file = output_path / f"{safe_name}.mar"

        # Check if already converted
        if mar_file.exists():
            print(f"[MAR] Using cached model: {mar_file}", flush=True)
            logger.info(f"Using cached MAR file: {mar_file}")
            return str(mar_file)

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"[MAR] Converting {model_name} to MAR format...", flush=True)
            logger.info(f"Converting {model_name} to MAR format...")

            # Create temp directory for model files
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Save model and tokenizer
                print(f"[MAR] Loading model and tokenizer...", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)

                model_dir = tmp_path / "model"
                model_dir.mkdir()

                print(f"[MAR] Saving model files...", flush=True)
                tokenizer.save_pretrained(str(model_dir))
                model.save_pretrained(str(model_dir))

                # Create handler file
                handler_file = tmp_path / "handler.py"
                handler_file.write_text(self._get_handler_code(handler))

                # Create MAR using torch-model-archiver
                print(f"[MAR] Creating MAR archive...", flush=True)
                self._create_mar_archive(
                    model_name=safe_name,
                    model_dir=str(model_dir),
                    handler_file=str(handler_file),
                    output_path=str(output_path),
                )

            print(f"[MAR] Conversion complete: {mar_file}", flush=True)
            logger.info(f"MAR file saved to: {mar_file}")
            return str(mar_file)

        except ImportError as e:
            raise ImportError(f"Missing dependency for MAR conversion: {e}")
        except Exception as e:
            if mar_file.exists():
                mar_file.unlink()
            raise RuntimeError(f"Failed to convert to MAR: {e}")

    def _get_handler_code(self, handler_type: str) -> str:
        """Generate TorchServe handler code."""
        return '''"""
TorchServe Handler for Text Generation
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class TextGenerationHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, data):
        texts = []
        for row in data:
            if isinstance(row, dict):
                text = row.get("data") or row.get("body") or row.get("text", "")
            else:
                text = row.decode("utf-8") if isinstance(row, bytes) else str(row)
            texts.append(text)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return outputs

    def postprocess(self, outputs):
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append({"generated_text": text})
        return results
'''

    def _create_mar_archive(
        self,
        model_name: str,
        model_dir: str,
        handler_file: str,
        output_path: str,
    ) -> None:
        """Create MAR archive using torch-model-archiver."""
        import subprocess

        cmd = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", "1.0",
            "--serialized-file", os.path.join(model_dir, "pytorch_model.bin"),
            "--handler", handler_file,
            "--extra-files", f"{model_dir}/config.json,{model_dir}/tokenizer.json,{model_dir}/vocab.json",
            "--export-path", output_path,
            "--force",
        ]

        # Check if pytorch_model.bin exists, otherwise use safetensors
        if not os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
            # Try to find model weights file
            model_files = list(Path(model_dir).glob("*.safetensors"))
            if model_files:
                cmd[cmd.index("--serialized-file") + 1] = str(model_files[0])
            else:
                # No serialized file, use handler-only mode
                cmd = [c for c in cmd if c not in ["--serialized-file", cmd[cmd.index("--serialized-file") + 1]]]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"torch-model-archiver failed: {result.stderr}")

    def create_triton_config(
        self,
        model_name: str,
        onnx_path: str,
        output_dir: str,
        max_batch_size: int = 8,
        platform: str = "onnxruntime_onnx",
    ) -> str:
        """
        Create Triton model repository structure.

        Args:
            model_name: Model name for Triton
            onnx_path: Path to ONNX model file
            output_dir: Output directory for Triton model repository
            max_batch_size: Maximum batch size
            platform: Triton backend platform

        Returns:
            Path to model directory in repository
        """
        safe_name = model_name.replace("/", "_")
        model_dir = Path(output_dir) / safe_name
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy ONNX model
        onnx_src = Path(onnx_path) / "model.onnx"
        if onnx_src.exists():
            shutil.copy(onnx_src, version_dir / "model.onnx")

        # Create config.pbtxt
        config_content = f'''name: "{safe_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [-1]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [-1]
  }}
]

output [
  {{
    name: "logits"
    data_type: TYPE_FP32
    dims: [-1, -1]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
  }}
]

dynamic_batching {{
  preferred_batch_size: [1, 2, 4, 8]
  max_queue_delay_microseconds: 100
}}
'''

        config_file = model_dir / "config.pbtxt"
        config_file.write_text(config_content)

        logger.info(f"Triton model repository created: {model_dir}")
        return str(model_dir)

"""Model export for deployment.

Exports trained models to various formats:
  - GGUF (for llama.cpp / LM Studio)
  - ONNX (for ONNX Runtime / NPU)
  - Merged (standalone HF model)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.config import Config, load
from src.lfm3_support import patch_transformers_for_lfm3, is_lfm3_model
from src.locking import atomic_write_json


@dataclass
class ExportResult:
    """Result of a model export."""
    format: str
    output_path: str
    size_bytes: int = 0
    success: bool = False
    error: str | None = None
    duration_seconds: float = 0.0


def export_merged(checkpoint_path: str, output_path: str,
                  base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> ExportResult:
    """Merge LoRA adapter into a standalone model."""
    result = ExportResult(format="merged", output_path=output_path)
    start = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch

        # Apply LFM3 patches if needed
        if is_lfm3_model(base_model):
            patch_transformers_for_lfm3()

        os.makedirs(output_path, exist_ok=True)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Load and merge adapter
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

        # Save merged model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        result.success = True
        result.size_bytes = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if os.path.isfile(os.path.join(output_path, f))
        )
    except Exception as e:
        result.error = str(e)
    result.duration_seconds = time.time() - start
    return result


def export_gguf(checkpoint_path: str, output_path: str,
                quantize: str = "q8_0") -> ExportResult:
    """Export model to GGUF format (requires llama.cpp)."""
    result = ExportResult(format=f"gguf-{quantize}", output_path=output_path)
    start = time.time()
    try:
        # Check for llama.cpp conversion script
        convert_script = None
        for path in ["/usr/local/bin/convert_hf_to_gguf.py",
                     "/opt/llama.cpp/convert_hf_to_gguf.py",
                     os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")]:
            if os.path.exists(path):
                convert_script = path
                break

        if convert_script is None:
            result.error = "llama.cpp convert script not found"
            return result

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # First merge if needed
        merged_dir = output_path + ".merged"
        merge_result = export_merged(checkpoint_path, merged_dir)
        if not merge_result.success:
            result.error = f"Merge failed: {merge_result.error}"
            return result

        # Convert to GGUF
        cmd = [sys.executable, convert_script, merged_dir,
               "--outfile", output_path, "--outtype", quantize]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            result.error = proc.stderr[-500:]
            return result

        result.success = True
        if os.path.exists(output_path):
            result.size_bytes = os.path.getsize(output_path)

        # Cleanup merged dir
        import shutil
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)

    except Exception as e:
        result.error = str(e)
    result.duration_seconds = time.time() - start
    return result


def export_onnx(checkpoint_path: str, output_path: str,
                base_model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> ExportResult:
    """Export model to ONNX format."""
    result = ExportResult(format="onnx", output_path=output_path)
    start = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # Apply LFM3 patches if needed
        if is_lfm3_model(base_model):
            patch_transformers_for_lfm3()

        os.makedirs(output_path, exist_ok=True)

        # Load and merge
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Export to ONNX
        dummy_input = tokenizer("Hello", return_tensors="pt")
        torch.onnx.export(
            model,
            (dummy_input["input_ids"],),
            os.path.join(output_path, "model.onnx"),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                          "logits": {0: "batch", 1: "seq"}},
            opset_version=14,
        )
        tokenizer.save_pretrained(output_path)

        result.success = True
        result.size_bytes = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
            if os.path.isfile(os.path.join(output_path, f))
        )
    except Exception as e:
        result.error = str(e)
    result.duration_seconds = time.time() - start
    return result


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Model export")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--format", choices=["merged", "gguf", "onnx"], default="merged")
    parser.add_argument("--quantize", default="q8_0", help="GGUF quantization type")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args(argv)

    if args.format == "merged":
        result = export_merged(args.checkpoint, args.output, args.base_model)
    elif args.format == "gguf":
        result = export_gguf(args.checkpoint, args.output, args.quantize)
    elif args.format == "onnx":
        result = export_onnx(args.checkpoint, args.output, args.base_model)

    if result.success:
        print(f"[export] Success: {result.format} -> {result.output_path} "
              f"({result.size_bytes / 1e6:.1f}MB, {result.duration_seconds:.1f}s)")
    else:
        print(f"[export] Failed: {result.error}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

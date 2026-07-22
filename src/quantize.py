"""Post-merge quantization for faster inference.

Quantizes merged models to GPTQ/AWQ format for smaller artifacts
and faster inference on GPU/CPU.

Usage:
    python -m src.cli quantize --label=<name> --bits=4
    python -m src.cli quantize-status
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import Config


@dataclass
class QuantizeResult:
    """Result of a quantization operation."""
    success: bool
    label: str
    source_path: str
    output_path: str
    bits: int
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    duration_seconds: float
    message: str


def _get_dir_size_mb(path: str) -> float:
    """Get directory size in MB."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    dataset_size: int = 128,
) -> QuantizeResult:
    """Quantize a model using GPTQ (requires auto-gptq)."""
    start = time.time()
    label = os.path.basename(model_path).replace("toolcall-v5-3b-", "").replace("-merged", "")

    original_size = _get_dir_size_mb(model_path)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        print(f"[quantize] loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=True,
        )

        print(f"[quantize] quantizing to {bits}-bit GPTQ...")
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path, quantize_config, trust_remote_code=True,
        )

        # Calibration data
        cal_data = []
        for text in [
            "Write a Python function to calculate fibonacci numbers.",
            "Explain how to use git for version control.",
            "Debug this error: ImportError: No module named 'foo'.",
        ]:
            cal_data.append(tokenizer(text, return_tensors="pt").input_ids)

        model.quantize(cal_data, batch_size=1)

        os.makedirs(output_path, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        quantized_size = _get_dir_size_mb(output_path)
        duration = time.time() - start

        return QuantizeResult(
            success=True, label=label, source_path=model_path,
            output_path=output_path, bits=bits,
            original_size_mb=original_size, quantized_size_mb=quantized_size,
            compression_ratio=original_size / max(quantized_size, 0.01),
            duration_seconds=duration,
            message=f"GPTQ {bits}-bit quantized",
        )

    except ImportError as e:
        return QuantizeResult(
            success=False, label=label, source_path=model_path,
            output_path="", bits=bits,
            original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, duration_seconds=time.time() - start,
            message=f"missing dependency: {e}",
        )
    except Exception as e:
        return QuantizeResult(
            success=False, label=label, source_path=model_path,
            output_path="", bits=bits,
            original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, duration_seconds=time.time() - start,
            message=f"quantization failed: {e}",
        )


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
) -> QuantizeResult:
    """Quantize a model using AWQ (requires autoawq)."""
    start = time.time()
    label = os.path.basename(model_path).replace("toolcall-v5-3b-", "").replace("-merged", "")

    original_size = _get_dir_size_mb(model_path)

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        print(f"[quantize] loading model: {model_path}")
        model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": bits}

        print(f"[quantize] quantizing to {bits}-bit AWQ...")
        model.quantize(tokenizer, quant_config=quant_config)

        os.makedirs(output_path, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        quantized_size = _get_dir_size_mb(output_path)
        duration = time.time() - start

        return QuantizeResult(
            success=True, label=label, source_path=model_path,
            output_path=output_path, bits=bits,
            original_size_mb=original_size, quantized_size_mb=quantized_size,
            compression_ratio=original_size / max(quantized_size, 0.01),
            duration_seconds=duration,
            message=f"AWQ {bits}-bit quantized",
        )

    except ImportError as e:
        return QuantizeResult(
            success=False, label=label, source_path=model_path,
            output_path="", bits=bits,
            original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, duration_seconds=time.time() - start,
            message=f"missing dependency: {e}",
        )
    except Exception as e:
        return QuantizeResult(
            success=False, label=label, source_path=model_path,
            output_path="", bits=bits,
            original_size_mb=original_size, quantized_size_mb=0,
            compression_ratio=0, duration_seconds=time.time() - start,
            message=f"quantization failed: {e}",
        )


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for quantize commands."""
    cmd = argv[1] if len(argv) > 1 else "quantize-status"

    label = None
    bits = 4
    method = "gptq"
    output_base = None

    for arg in argv:
        if arg.startswith("--label="):
            label = arg.split("=", 1)[1]
        elif arg.startswith("--bits="):
            bits = int(arg.split("=", 1)[1])
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif arg.startswith("--output="):
            output_base = arg.split("=", 1)[1]

    if cmd == "quantize":
        if not label:
            print("[error] quantize requires --label=<name>")
            return 2

        out_base = cfg.get("train", "output_dir",
                          default="/media/scott/data/finetune-staging/outputs/checkpoints")
        source = os.path.join(out_base, f"toolcall-v5-3b-{label}-merged")

        if not os.path.exists(source):
            print(f"[error] merged model not found: {source}")
            return 2

        if output_base is None:
            output_base = out_base

        output_path = os.path.join(output_base, f"toolcall-v5-3b-{label}-merged-{bits}bit")

        print(f"[quantize] {label} ({bits}-bit {method})")
        print(f"  source: {source}")
        print(f"  output: {output_path}")

        if method == "awq":
            result = quantize_awq(source, output_path, bits=bits)
        else:
            result = quantize_gptq(source, output_path, bits=bits)

        if result.success:
            print(f"[quantize] {result.message}")
            print(f"  {result.original_size_mb:.0f}MB -> {result.quantized_size_mb:.0f}MB "
                  f"({result.compression_ratio:.1f}x compression)")
            print(f"  {result.duration_seconds:.1f}s")
            return 0
        else:
            print(f"[quantize] FAILED: {result.message}")
            return 1

    if cmd == "quantize-status":
        out_base = cfg.get("train", "output_dir",
                          default="/media/scott/data/finetune-staging/outputs/checkpoints")

        quantized = []
        if os.path.isdir(out_base):
            for entry in os.listdir(out_base):
                if "merged" in entry and ("bit" in entry or "gptq" in entry or "awq" in entry):
                    path = os.path.join(out_base, entry)
                    size = _get_dir_size_mb(path)
                    quantized.append((entry, size))

        if not quantized:
            print("[quantize-status] no quantized models found")
            return 0

        print(f"[quantize-status] {len(quantized)} quantized models:")
        for name, size in sorted(quantized):
            print(f"  {name}: {size:.0f}MB")
        return 0

    print("Commands:")
    print("  quantize --label=<name> [--bits=4] [--method=gptq|awq]")
    print("  quantize-status")
    return 0

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


# Reduced fallback calibration set. GPTQ quality depends on the calibration set
# being representative; three prompts is far too few to approximate the model's
# activation distribution, so results built from this set are reported as such
# rather than being silently indistinguishable from a real calibration run.
BUILTIN_CALIBRATION_PROMPTS = (
    "Write a Python function to calculate fibonacci numbers.",
    "Explain how to use git for version control.",
    "Debug this error: ImportError: No module named 'foo'.",
)


def _row_text(row: object) -> str:
    """Flatten one dataset row to plain text for calibration."""
    if not isinstance(row, dict):
        return ""
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [m.get("content", "") for m in messages
                 if isinstance(m, dict) and isinstance(m.get("content"), str)]
        text = "\n".join(p for p in parts if p.strip())
        if text.strip():
            return text
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        parts = [m.get("value", "") for m in conversations
                 if isinstance(m, dict) and isinstance(m.get("value"), str)]
        text = "\n".join(p for p in parts if p.strip())
        if text.strip():
            return text
    return ""


def calibration_from_corpus(dataset_dir: str, label: str, limit: int) -> list[str]:
    """Pull calibration text from the built SFT corpus for this label.

    This is the distribution the model is actually being adapted to, so it is a
    far better calibration set than generic prompts. Returns [] if the corpus is
    missing so the caller can fall back and say so.
    """
    path = os.path.join(dataset_dir, f"train.{label}.jsonl")
    if not os.path.isfile(path):
        return []
    texts: list[str] = []
    try:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _row_text(row)
                if text:
                    texts.append(text)
                if len(texts) >= limit:
                    break
    except OSError:
        return []
    return texts


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    dataset_size: int = 128,
    calibration_texts: list[str] | None = None,
) -> QuantizeResult:
    """Quantize a model using GPTQ (requires auto-gptq).

    calibration_texts, when supplied, is truncated to dataset_size and used as
    the calibration set. When omitted the built-in prompt set is used and the
    result says so, because a 3-sample calibration produces a measurably worse
    model than a representative one and that must not be invisible.
    """
    start = time.time()
    label = os.path.basename(model_path).removeprefix("toolcall-v5-3b-").removesuffix("-merged")

    original_size = _get_dir_size_mb(model_path)

    texts = list(calibration_texts or [])[:dataset_size] or list(BUILTIN_CALIBRATION_PROMPTS)
    from_corpus = bool(calibration_texts)

    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer

        print(f"[quantize] loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=True,
        )

        print(f"[quantize] quantizing to {bits}-bit GPTQ with "
              f"{len(texts)} calibration samples "
              f"({'corpus' if from_corpus else 'BUILTIN FALLBACK'})...")
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path, quantize_config, trust_remote_code=True,
        )

        cal_data = []
        for text in texts:
            cal_data.append(tokenizer(text, return_tensors="pt").input_ids)

        model.quantize(cal_data, batch_size=1)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        quantized_size = _get_dir_size_mb(output_path)
        duration = time.time() - start

        message = f"GPTQ {bits}-bit quantized"
        if not from_corpus:
            message += (f" (WARNING: reduced {len(texts)}-sample built-in "
                        f"calibration set; pass calibration_texts for a "
                        f"representative one)")
        else:
            message += f" ({len(texts)} corpus calibration samples)"

        return QuantizeResult(
            success=True, label=label, source_path=model_path,
            output_path=output_path, bits=bits,
            original_size_mb=original_size, quantized_size_mb=quantized_size,
            compression_ratio=original_size / max(quantized_size, 0.01),
            duration_seconds=duration,
            message=message,
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
    """Quantize a model using AWQ (requires autoawq).

    AWQ derives its own calibration from the tokenizer and does not accept a
    caller-supplied set, so there is no dataset_size equivalent here.
    """
    start = time.time()
    label = os.path.basename(model_path).removeprefix("toolcall-v5-3b-").removesuffix("-merged")

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

        # Clear any previous/partial output so a failed run cannot leave a
        # half-written directory that quantize-status then reports as valid.
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
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
    dataset_size = 128

    for arg in argv:
        if arg.startswith("--label="):
            label = arg.split("=", 1)[1]
        elif arg.startswith("--bits="):
            raw_bits = arg.split("=", 1)[1]
            # Unvalidated int() here raised a bare ValueError traceback.
            try:
                bits = int(raw_bits)
            except ValueError:
                print(f"[error] --bits must be an integer, got {raw_bits!r}")
                return 2
        elif arg.startswith("--calibration-size="):
            raw_size = arg.split("=", 1)[1]
            try:
                dataset_size = int(raw_size)
            except ValueError:
                print(f"[error] --calibration-size must be an integer, got {raw_size!r}")
                return 2
            if dataset_size <= 0:
                print("[error] --calibration-size must be positive")
                return 2
        elif arg.startswith("--method="):
            method = arg.split("=", 1)[1]
        elif arg.startswith("--output="):
            output_base = arg.split("=", 1)[1]

    if method not in ("gptq", "awq"):
        print(f"[error] --method must be gptq or awq, got {method!r}")
        return 2

    if cmd == "quantize":
        if not label:
            print("[error] quantize requires --label=<name>")
            return 2
        if bits not in (2, 3, 4, 8):
            print(f"[error] --bits must be one of 2, 3, 4, 8 (got {bits}); "
                  f"other widths are not supported by GPTQ/AWQ")
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
            # Calibrate on the real SFT corpus for this label when it exists.
            dataset_dir = cfg.get("paths", "dataset_dir",
                                  default="/media/scott/data/finetune-staging/data/datasets")
            corpus = calibration_from_corpus(str(dataset_dir), label, dataset_size)
            if corpus:
                print(f"  calibration: {len(corpus)} samples from "
                      f"{os.path.join(str(dataset_dir), f'train.{label}.jsonl')}")
            else:
                print(f"  calibration: corpus not found for label {label!r} "
                      f"in {dataset_dir}; falling back to the reduced built-in set")
            result = quantize_gptq(source, output_path, bits=bits,
                                    dataset_size=dataset_size,
                                    calibration_texts=corpus or None)

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

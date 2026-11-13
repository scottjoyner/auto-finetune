"""Auto-scaling for training based on available VRAM and system resources.

Dynamically adjusts batch size, sequence length, and gradient accumulation
to maximize throughput while staying within VRAM limits.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.config import Config, load


@dataclass
class ScalingConfig:
    """Recommended scaling parameters."""
    max_seq_length: int = 64
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    load_in_4bit: bool = False
    estimated_vram_gb: float = 0.0
    available_vram_gb: float = 0.0
    throughput_estimate: float = 0.0


def detect_vram() -> tuple[float, float]:
    """Detect total and available VRAM in GB.

    Returns (total_gb, available_gb).
    """
    # Try ROCm
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for key, val in data.items():
                if isinstance(val, dict):
                    total = float(val.get("VRAM Total (B)", 0) or val.get("Total VRAM", 0))
                    used = float(val.get("VRAM Used (B)", 0) or val.get("Used VRAM", 0))
                    if math.isfinite(total) and math.isfinite(used) and total > 0 and 0 <= used <= total:
                        total_gb = total / (1024**3)
                        avail_gb = (total - used) / (1024**3)
                        return total_gb, avail_gb
    except Exception:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                total_mb = float(parts[0].strip())
                free_mb = float(parts[1].strip())
                if all(math.isfinite(n) for n in (total_mb, free_mb)) and total_mb > 0 and 0 <= free_mb <= total_mb:
                    return total_mb / 1024, free_mb / 1024
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            # Do not invent free capacity from a fraction of total VRAM.
            free_b, total_b = torch.cuda.mem_get_info(0)
            if total_b > 0 and 0 <= free_b <= total_b:
                return total_b / (1024**3), free_b / (1024**3)
    except Exception:
        pass

    # Unknown telemetry is a resource HOLD; synthetic 15/8 GB was unsafe.
    return 0.0, 0.0


def recommend(available_vram_gb: float | None = None,
              total_vram_gb: float | None = None,
              model_params_b: float = 0.5,
              prefer_quality: bool = False,
              model_name: str = "") -> ScalingConfig:
    """Recommend scaling parameters based on available VRAM.

    Args:
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        total_vram_gb: Total VRAM in GB (auto-detected if None)
        model_params_b: Model size in billions of parameters
        prefer_quality: If True, prefer longer sequences over larger batches
        model_name: Model name for model-specific recommendations (e.g. LFM3-1.7B)
    """
    """Recommend scaling parameters based on available VRAM.

    Args:
        available_vram_gb: Available VRAM in GB (auto-detected if None)
        total_vram_gb: Total VRAM in GB (auto-detected if None)
        model_params_b: Model size in billions of parameters
        prefer_quality: If True, prefer longer sequences over larger batches
    """
    if available_vram_gb is None or total_vram_gb is None:
        total, avail = detect_vram()
        if available_vram_gb is None:
            available_vram_gb = avail
        if total_vram_gb is None:
            total_vram_gb = total

    if (not math.isfinite(total_vram_gb) or not math.isfinite(available_vram_gb)
            or total_vram_gb <= 0 or available_vram_gb <= 0
            or available_vram_gb > total_vram_gb):
        raise ValueError("measured_vram_unavailable")
    if not math.isfinite(model_params_b) or model_params_b <= 0:
        raise ValueError("invalid_model_size")
    # Conservative bf16 training heuristic, NOT authorization for a GPU lease.
    # Do not recommend an executable config when even the baseline model state
    # and 1 GiB headroom exceed physically measured free VRAM.
    model_mem_gb = model_params_b * 2 * 3
    if available_vram_gb < model_mem_gb + 1:
        raise ValueError("insufficient_vram_for_model")

    config = ScalingConfig(
        estimated_vram_gb=total_vram_gb,
        available_vram_gb=available_vram_gb,
    )

    # LFM3-1.7B specific: hybrid conv/attention, fits in bf16 on 15GB VRAM
    if "lfm3" in model_name.lower() and model_params_b < 2:
        config.max_seq_length = 4096
        config.per_device_batch_size = 1
        config.gradient_accumulation_steps = 16
        config.load_in_4bit = False
        config.throughput_estimate = 2.0
        return config

    # model_mem_gb was checked above as an admission-independent heuristic.
    # Available for data/activations
    free_for_data = available_vram_gb - model_mem_gb - 1.0  # 1GB headroom
    if free_for_data < 0:
        free_for_data = available_vram_gb * 0.3

    # Sequence length vs batch size tradeoff
    # Memory per example ≈ seq_len * hidden_dim * 4 bytes (activations)
    hidden_dim = 896 if model_params_b < 1 else 4096  # rough estimate

    if prefer_quality:
        # Prefer longer sequences
        if free_for_data > 4:
            config.max_seq_length = 512
            config.per_device_batch_size = 1
        elif free_for_data > 2:
            config.max_seq_length = 256
            config.per_device_batch_size = 1
        elif free_for_data > 1:
            config.max_seq_length = 128
            config.per_device_batch_size = 1
        else:
            config.max_seq_length = 64
            config.per_device_batch_size = 1
    else:
        # Prefer larger batches
        if free_for_data > 4:
            config.max_seq_length = 128
            config.per_device_batch_size = 4
            config.gradient_accumulation_steps = 2
        elif free_for_data > 2:
            config.max_seq_length = 128
            config.per_device_batch_size = 2
            config.gradient_accumulation_steps = 4
        elif free_for_data > 1:
            config.max_seq_length = 64
            config.per_device_batch_size = 1
            config.gradient_accumulation_steps = 8
        else:
            config.max_seq_length = 32
            config.per_device_batch_size = 1
            config.gradient_accumulation_steps = 16

    # 4-bit for large models or low VRAM
    if model_params_b > 3 or available_vram_gb < 8:
        config.load_in_4bit = True

    # Throughput estimate
    config.throughput_estimate = (config.per_device_batch_size *
                                  config.gradient_accumulation_steps) / max(1, config.max_seq_length / 64)

    return config


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Auto-scaling recommendation")
    parser.add_argument("--model-size", type=float, default=0.5, help="Model size in billions")
    parser.add_argument("--prefer-quality", action="store_true", help="Prefer longer sequences")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    try:
        config = recommend(model_params_b=args.model_size, prefer_quality=args.prefer_quality)
    except ValueError as exc:
        print(f"[autoscale] HOLD: {exc}")
        return 78

    if args.json:
        import json as j
        print(j.dumps(asdict(config), indent=2))
    else:
        print(f"[autoscale] VRAM: {config.available_vram_gb:.1f}GB available / {config.estimated_vram_gb:.1f}GB total")
        print(f"[autoscale] Recommended:")
        print(f"  max_seq_length: {config.max_seq_length}")
        print(f"  batch_size: {config.per_device_batch_size}")
        print(f"  grad_accum: {config.gradient_accumulation_steps}")
        print(f"  4-bit: {config.load_in_4bit}")
        print(f"  throughput estimate: {config.throughput_estimate:.2f}")
        print()
        print("Env vars:")
        print(f"  TRAIN_MAX_SEQ_LENGTH={config.max_seq_length}")
        print(f"  TRAIN_PER_DEVICE_BATCH_SIZE={config.per_device_batch_size}")
        print(f"  TRAIN_GRADIENT_ACCUMULATION_STEPS={config.gradient_accumulation_steps}")
        print(f"  TRAIN_LOAD_4BIT={'1' if config.load_in_4bit else '0'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

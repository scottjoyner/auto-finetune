"""Auto-scaling for training based on available VRAM and system resources.

Dynamically adjusts batch size, sequence length, and gradient accumulation
to maximize throughput while staying within VRAM limits.
"""
from __future__ import annotations

import json
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
                    total = val.get("VRAM Total (B)", 0) or val.get("Total VRAM", 0)
                    used = val.get("VRAM Used (B)", 0) or val.get("Used VRAM", 0)
                    if total:
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
                return total_mb / 1024, free_mb / 1024
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Estimate available as 80% of total (conservative)
            return total, total * 0.8
    except Exception:
        pass

    # Fallback
    return 15.0, 8.0


def recommend(available_vram_gb: float | None = None,
              total_vram_gb: float | None = None,
              model_params_b: float = 0.5,
              prefer_quality: bool = False) -> ScalingConfig:
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

    config = ScalingConfig(
        estimated_vram_gb=total_vram_gb,
        available_vram_gb=available_vram_gb,
    )

    # Model memory estimate (bf16): params * 2 bytes + optimizer (2x) + gradients
    model_mem_gb = model_params_b * 2 * 3  # weights + grad + optimizer state

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

    config = recommend(model_params_b=args.model_size, prefer_quality=args.prefer_quality)

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

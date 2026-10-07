"""Bounded, synthetic-only activation-perturbation LoRA experiment.

No pretrained model, corpus, credential, optimizer, scheduler or checkpoint accessed.
This probes Dust-inspired tokenwise forward-only estimation, NOT upstream Dust
reproduction or an implemented K2-Horizon training backend.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


def run(*, device: str = "cpu", seed: int = 42, draws: int = 128,
        steps: int = 6, sigma: float = 0.05, lr: float = 2.0) -> dict:
    if not 1 <= draws <= 512 or not 1 <= steps <= 20:
        raise ValueError("Bounded probe requires draws in [1,512] and steps in [1,20]")
    if not 0 < sigma <= 1.0 or not 0 < lr <= 5.0:
        raise ValueError("Invalid perturbation scale or learning rate")
    if device not in ("cpu", "cuda"):
        raise ValueError("Only cpu or cuda device alias supported")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device alias unavailable; ROCm uses the same alias")

    dev = torch.device(device)
    torch.manual_seed(seed)
    cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
    noise_rng = torch.Generator(device=dev).manual_seed(seed + 1000)
    n, din, dout, rank = 128, 32, 12, 4
    alpha = rank
    scale = alpha / rank
    with torch.no_grad():
        x = torch.randn(n, din, generator=cpu_rng).to(dev)
        base = (torch.randn(dout, din, generator=cpu_rng) / math.sqrt(din)).to(dev)
        a = (torch.randn(rank, din, generator=cpu_rng) / math.sqrt(din)).to(dev)
        teacher_b = (torch.randn(dout, rank, generator=cpu_rng) * 0.5).to(dev)
        b = torch.zeros_like(teacher_b)
        target = x @ base.T + scale * (x @ a.T) @ teacher_b.T
        history = []
        cosines = []
        t0 = time.monotonic()
        for step in range(steps):
            z = x @ a.T
            y = x @ base.T + scale * z @ b.T
            per_token = (y - target).square().mean(-1) * 0.5
            current_loss = per_token.mean().item()
            history.append(current_loss)
            directions = torch.randn(
                draws, n, dout, device=dev, generator=noise_rng, dtype=torch.float32
            )
            y_noisy = y.unsqueeze(0) + sigma * directions
            noisy_loss = (y_noisy - target.unsqueeze(0)).square().mean(-1) * 0.5
            reward = per_token.unsqueeze(0) - noisy_loss
            if draws > 1:
                reward -= reward.mean(dim=0, keepdim=True)
            estimated_dy = -(reward.unsqueeze(-1) * directions).mean(0) / sigma
            grad_b = scale * estimated_dy.T @ z / n
            grad_a = scale * (estimated_dy @ b).T @ x / n
            reference = scale * ((y - target) / dout).T @ z / n
            cosine = torch.nn.functional.cosine_similarity(
                grad_b.flatten(), reference.flatten(), dim=0, eps=1e-12
            ).item()
            cosines.append(cosine)
            b.add_(grad_b, alpha=-lr)
            a.add_(grad_a, alpha=-lr)
        final_loss = ((x @ base.T + scale * (x @ a.T) @ b.T - target).square()
                      .mean(-1) * 0.5).mean().item()
        history.append(final_loss)
        if not all(math.isfinite(v) for v in history + cosines):
            raise ArithmeticError("Nonfinite training metric")
        if any(t.requires_grad or t.grad is not None for t in (x, base, a, b)):
            raise AssertionError("Probe requires no autograd and no parameter gradients")
        elapsed = time.monotonic() - t0
    return {
        "kind": "synthetic_only_forward_zeroth_order_lora_probe",
        "not_k2_training": True,
        "device": device, "torch_version": torch.__version__,
        "torch_hip": torch.version.hip, "seed": seed, "draws": draws,
        "steps": steps, "sigma": sigma, "learning_rate": lr,
        "initial_loss": history[0], "final_loss": final_loss,
        "loss_history": history, "gradient_cosines": cosines,
        "duration_seconds": elapsed,
        "improved_on_training_samples": final_loss < history[0],
        "backward_calls": 0, "checkpoint_written": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--output", type=Path, help="Exclusively create optional JSON evidence")
    args = parser.parse_args(argv)
    report = run(device=args.device, seed=args.seed, draws=args.draws,
                 steps=args.steps, sigma=args.sigma, lr=args.lr)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8") as out:
                out.write(encoded)
        except FileExistsError:
            parser.error("Refusing to overwrite existing evidence")
    return 0 if report["improved_on_training_samples"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

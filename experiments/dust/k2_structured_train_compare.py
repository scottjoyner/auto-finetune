"""Matched K2 LoRA training: exact backprop vs orthogonal antithetic forward-only updates.

Research-only follow-up to gradient calibration. Uses fresh rank-4 final-o_proj
adapters, identical initialization and train order, frozen pretrained base, and
a prompt-disjoint heldout slice. No checkpoint is written or promoted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from experiments.dust.k2_data import select_disjoint
from experiments.dust.k2_forward_only import FinalProjectionLoRA
from experiments.dust.k2_matched_compare import (
    backprop_step, digest, load_base, loss_on_heldout, read_mem_available,
    score_one,
)


def cosine_error(a, b) -> dict:
    import torch
    x = a.detach().float().flatten()
    y = b.detach().float().flatten()
    ny = torch.linalg.vector_norm(y)
    nx = torch.linalg.vector_norm(x)
    if float(ny) <= 1e-20:
        return {"cosine": None, "relative_l2_error": None,
                "norm_ratio": None, "exact_update_zero": True}
    cosine = torch.dot(x, y) / (torch.clamp(nx, min=1e-20) * ny)
    return {
        "cosine": float(cosine.item()),
        "relative_l2_error": float((torch.linalg.vector_norm(x-y)/ny).item()),
        "norm_ratio": float((nx/ny).item()),
        "exact_update_zero": False,
    }


def structured_step(model, adapter: FinalProjectionLoRA, sample: dict, device: str,
                    *, seed: int, population: int, sigma: float,
                    lr: float) -> dict:
    import torch
    if population > adapter.b.shape[0]:
        raise ValueError("Orthogonal population cannot exceed output hidden size")
    with torch.no_grad():
        adapter.jitter = None
        clean, mask = score_one(model, sample, device)
        if not bool(torch.isfinite(clean).all()):
            raise ArithmeticError("Nonfinite clean training CE")
        hidden = adapter.b.shape[0]
        gen = torch.Generator(device=device).manual_seed(seed)
        raw = torch.randn(hidden, population, generator=gen,
                          device=device, dtype=torch.float32)
        directions = (
            torch.linalg.qr(raw, mode="reduced").Q.T * math.sqrt(hidden)
        ).contiguous()
        del raw
        estimate_sum = torch.zeros(
            (*clean.shape, hidden), device=device, dtype=torch.float32)
        for index in range(population):
            if read_mem_available() < 1536 * 1024**2:
                raise MemoryError("Host memory below structured-step floor")
            noise = directions[index].view(1, 1, hidden).expand_as(estimate_sum)
            adapter.jitter = sigma * noise
            try:
                plus, plus_mask = score_one(model, sample, device)
                adapter.jitter = -sigma * noise
                minus, minus_mask = score_one(model, sample, device)
            finally:
                adapter.jitter = None
            if not torch.equal(mask, plus_mask) or not torch.equal(mask, minus_mask):
                raise AssertionError("Causal mask changed under perturbation")
            valid = mask.float()
            estimate_sum += (
                ((plus.float()-minus.float())*valid).unsqueeze(-1)
                * noise / (2.0*sigma*population)
            )
        adapter.apply(estimate_sum, count=int(mask.sum().item()), lr=lr)
        return {
            "tokens": int(mask.sum().item()),
            "train_ce": float(clean.sum().item()/mask.sum().item()),
            "population": population, "sigma": sigma,
            "forward_calls": 1 + 2*population,
        }


def run_backend(model, *, backend: str, train: list[dict], heldout: list[dict],
                device: str, seed: int, steps: int, population: int,
                sigma: float, lr: float, max_seconds: int):
    import torch
    started = time.monotonic()
    adapter = FinalProjectionLoRA(model, rank=4, seed=seed + 17)
    original_weight = adapter.module.weight.detach().clone()
    initial_a = adapter.a.detach().clone()
    initial_b = adapter.b.detach().clone()
    adapter.a.requires_grad_(backend == "backprop")
    adapter.b.requires_grad_(backend == "backprop")
    history = []
    try:
        pre_train = loss_on_heldout(model, train, device)
        pre_eval = loss_on_heldout(model, heldout, device)
        for step in range(steps):
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("Backend exceeded wall-clock budget")
            row = train[step % len(train)]
            if backend == "backprop":
                one = backprop_step(model, adapter, row, device, lr=lr)
                one["forward_calls"] = 1
            elif backend == "orthogonal_antithetic":
                one = structured_step(
                    model, adapter, row, device,
                    seed=seed*100000 + step,
                    population=population, sigma=sigma, lr=lr)
            else:
                raise ValueError("Unknown backend")
            if device == "cuda":
                torch.cuda.synchronize()
            history.append({"step": step + 1, **one,
                            "elapsed_seconds": time.monotonic()-started})
        post_train = loss_on_heldout(model, train, device)
        post_eval = loss_on_heldout(model, heldout, device)
        final_a = adapter.a.detach().clone()
        final_b = adapter.b.detach().clone()
        if not torch.equal(original_weight, adapter.module.weight):
            raise AssertionError("Frozen base projection changed")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Frozen base accumulated gradients")
        return {
            "metrics": {
                "backend": backend,
                "train_before": pre_train, "train_after": post_train,
                "heldout_before": pre_eval, "heldout_after": post_eval,
                "train_ce_delta": post_train["mean_token_ce"] - pre_train["mean_token_ce"],
                "heldout_ce_delta": post_eval["mean_token_ce"] - pre_eval["mean_token_ce"],
                "history": history,
                "elapsed_seconds": time.monotonic()-started,
                "base_unchanged": True, "base_model_grads_absent": True,
                "backward_calls": steps if backend == "backprop" else 0,
                "checkpoint_written": False,
            },
            "initial_a": initial_a, "initial_b": initial_b,
            "final_a": final_a, "final_b": final_b,
        }
    finally:
        adapter.close()


def run_compare(*, model_dir: Path, train_file: Path, heldout_file: Path,
                expected_sha: str, device: str, seed: int,
                train_count: int, eval_count: int, steps: int,
                population: int, sigma: float, lr: float,
                train_max_tokens: int, eval_max_tokens: int,
                max_seconds: int) -> dict:
    import torch
    from transformers import AutoTokenizer
    if not 2 <= steps <= 8:
        raise ValueError("Structured comparison requires 2..8 steps")
    if population not in (256, 512, 1024, 1536):
        raise ValueError("Population must be one of 256/512/1024/1536")
    if not (0 < sigma <= .25 and 0 < lr <= 1.0):
        raise ValueError("Invalid sigma or learning rate")
    if train_file.resolve() == heldout_file.resolve():
        raise ValueError("Train and heldout sources must differ")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    train, heldout, dataset = select_disjoint(
        train_file, heldout_file, tokenizer,
        train_count=train_count, eval_count=eval_count,
        max_tokens=train_max_tokens, eval_max_tokens=eval_max_tokens)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, model_sha = load_base(model_dir, device=device, expected_sha=expected_sha)
    try:
        exact = run_backend(
            model, backend="backprop", train=train, heldout=heldout,
            device=device, seed=seed, steps=steps, population=population,
            sigma=sigma, lr=lr, max_seconds=max_seconds)
        structured = run_backend(
            model, backend="orthogonal_antithetic", train=train, heldout=heldout,
            device=device, seed=seed, steps=steps, population=population,
            sigma=sigma, lr=lr, max_seconds=max_seconds)
        a_exact_update = exact["final_a"] - exact["initial_a"]
        b_exact_update = exact["final_b"] - exact["initial_b"]
        a_structured_update = structured["final_a"] - structured["initial_a"]
        b_structured_update = structured["final_b"] - structured["initial_b"]
        return {
            "schema": "auto-finetune.dust-k2-structured-train.v1",
            "research_only": True, "production_promotion_authorized": False,
            "model_weights_sha256": model_sha,
            "model_config_sha256": digest(model_dir/"config.json"),
            "device": device, "torch": torch.__version__, "hip": torch.version.hip,
            "seed": seed, "rank": 4, "steps": steps,
            "population": population, "sigma": sigma, "learning_rate": lr,
            "dataset": dataset,
            "backprop": exact["metrics"],
            "structured": structured["metrics"],
            "adapter_update_alignment": {
                "a": cosine_error(a_structured_update, a_exact_update),
                "b": cosine_error(b_structured_update, b_exact_update),
            },
            "max_device_memory_allocated": (
                torch.cuda.max_memory_allocated() if device=="cuda" else None),
            "remaining_host_mem_available_bytes": read_mem_available(),
            "raw_user_data_in_report": False,
            "checkpoint_written": False,
            "warning": (
                "Tiny final-layer research comparison only; no claim of "
                "generalization, production readiness or upstream Dust parity."),
        }
    finally:
        del model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-jsonl", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cuda")
    parser.add_argument("--expected-sha256", default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=6)
    parser.add_argument("--train-max-tokens", type=int, default=128)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=.25)
    parser.add_argument("--lr", type=float, default=.1)
    parser.add_argument("--max-seconds", type=int, default=540)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    if not 1 <= args.train_count <= 24 or not 1 <= args.eval_count <= 12:
        parser.error("Sample count outside research bounds")
    if not 30 <= args.max_seconds <= 900:
        parser.error("max-seconds outside [30,900]")
    result=run_compare(
        model_dir=args.model_dir, train_file=args.train_jsonl,
        heldout_file=args.heldout_jsonl, expected_sha=args.expected_sha256,
        device=args.device, seed=args.seed, train_count=args.train_count,
        eval_count=args.eval_count, steps=args.steps, population=args.population,
        sigma=args.sigma, lr=args.lr, train_max_tokens=args.train_max_tokens,
        eval_max_tokens=args.eval_max_tokens, max_seconds=args.max_seconds)
    rendered=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output is None:
        print(rendered,end="")
    else:
        with args.output.open("x",encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__=="__main__":
    raise SystemExit(main())

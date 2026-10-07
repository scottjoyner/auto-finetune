"""Controlled K2-Horizon real-weight Dust-inspired vs matched backprop LoRA pilot.

Research only: two fresh final-o_proj rank-4 adapters, same seeded initialization,
training examples, immutable heldout prompts and BF16 model. No merge or save.
Not an upstream Dust reproduction, nor deployment-eligible model evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from experiments.dust.k2_data import select_disjoint
from experiments.dust.k2_forward_only import FinalProjectionLoRA, position_losses


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot establish system memory headroom")


def load_base(directory: Path, *, device: str, expected_sha: str):
    import torch
    from transformers import AutoModelForCausalLM
    target = directory / "model-00000-of-00001.safetensors"
    if not target.is_file():
        raise FileNotFoundError("Pinned K2 single-shard base weights absent")
    actual = digest(target)
    if actual != expected_sha:
        raise ValueError("Pinned base-model SHA256 mismatch")
    if read_mem_available() < 5 * 1024**3:
        raise MemoryError("Xwing system memory below 5 GiB before base load")
    if device == "cuda":
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("ROCm device is unavailable")
        if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
            raise MemoryError("Insufficient ROCm addressable free memory")
    # Local Transformers custom-code diagnostics can print to stdout;
    # reserve stdout for strictly valid JSON experiment evidence.
    with contextlib.redirect_stdout(sys.stderr):
        model = AutoModelForCausalLM.from_pretrained(
            str(directory), trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=device,
        ).eval().requires_grad_(False)
    if model.config.model_type != "k2_horizon" or model.config.num_hidden_layers != 28:
        raise ValueError("Not the expected K2-Horizon base architecture")
    return model, actual


def batch_to_tensors(sample: dict, device: str):
    import torch
    token_ids = torch.tensor([sample["tokens"]], device=device, dtype=torch.long)
    label_ids = torch.tensor([sample["labels"]], device=device, dtype=torch.long)
    return token_ids, label_ids


def score_one(model, sample: dict, device: str):
    inputs, labels = batch_to_tensors(sample, device)
    logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
    per_token, mask = position_losses(logits, labels)
    return per_token, mask


def loss_on_heldout(model, data: list[dict], device: str) -> dict:
    import torch
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for row in data:
            losses, mask = score_one(model, row, device)
            if not torch.isfinite(losses).all():
                raise ArithmeticError("Nonfinite heldout token CE")
            total_nll += float(losses.sum().item())
            total_tokens += int(mask.sum().item())
    value = total_nll / max(total_tokens, 1)
    return {"tokens": total_tokens, "mean_token_ce": value,
            "perplexity": math.exp(min(value, 80.0))}


def gpu_sync(device: str) -> None:
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def dust_step(model, adapter: FinalProjectionLoRA, sample: dict, device: str,
              *, seed: int, draws: int, sigma: float, lr: float) -> dict:
    import torch
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        adapter.jitter = None
        clean, mask = score_one(model, sample, device)
        if not torch.isfinite(clean).all():
            raise ArithmeticError("Nonfinite training token CE")
        hidden = adapter.b.shape[0]
        accumulated = torch.zeros(
            (*clean.shape, hidden), device=device, dtype=torch.float32)
        for _ in range(draws):
            if read_mem_available() < 1536 * 1024**2:
                raise MemoryError("Host system memory below research safety minimum")
            perturb = torch.randn(
                accumulated.shape, generator=generator, device=device,
                dtype=torch.float32)
            adapter.jitter = perturb * sigma
            try:
                altered, pert_mask = score_one(model, sample, device)
            finally:
                adapter.jitter = None
            if not torch.equal(mask, pert_mask):
                raise AssertionError("Invalid causal mask after perturbation")
            if not torch.isfinite(altered).all():
                raise ArithmeticError("Nonfinite perturbed token CE")
            reward = (clean - altered) * mask
            accumulated -= (reward.unsqueeze(-1) * perturb) / (sigma * draws)
        adapter.apply(accumulated, count=int(mask.sum().item()), lr=lr)
        return {"tokens": int(mask.sum()), "train_ce": float(
            clean.sum().item() / mask.sum().item())}


def backprop_step(model, adapter: FinalProjectionLoRA, sample: dict,
                  device: str, *, lr: float) -> dict:
    import torch
    with torch.enable_grad():
        adapter.jitter = None
        # Compute a gradient only for the fresh low-rank factors; all model
        # parameters remain frozen. This is the matched conventional control.
        losses, mask = score_one(model, sample, device)
        ce = losses.sum() / mask.sum()
        if not bool(torch.isfinite(ce)):
            raise ArithmeticError("Nonfinite backprop CE")
        ga, gb = torch.autograd.grad(ce, (adapter.a, adapter.b))
    if not all(bool(torch.isfinite(x).all()) for x in (ga, gb)):
        raise ArithmeticError("Nonfinite backprop update")
    with torch.no_grad():
        adapter.a.add_(ga, alpha=-lr)
        adapter.b.add_(gb, alpha=-lr)
    return {"tokens": int(mask.sum()), "train_ce": float(ce.detach().item())}


def one_backend(model, *, name: str, train: list[dict], eval_rows: list[dict],
                device: str, seed: int, steps: int, draws: int,
                sigma: float, lr: float, max_seconds: int) -> dict:
    import torch
    start = time.monotonic()
    adapter = FinalProjectionLoRA(model, rank=4, seed=seed + 17)
    original_weight = adapter.module.weight.detach().clone()
    initial_a = adapter.a.detach().clone()
    adapter.a.requires_grad_(name == "backprop")
    adapter.b.requires_grad_(name == "backprop")
    history = []
    try:
        pre = loss_on_heldout(model, eval_rows, device)
        for step in range(steps):
            if time.monotonic() - start > max_seconds:
                raise TimeoutError("Exceeded bounded experiment wall-clock budget")
            if read_mem_available() < 1536 * 1024**2:
                raise MemoryError("System memory pressure; experiment stopped")
            row = train[step % len(train)]
            if name == "dust":
                res = dust_step(model, adapter, row, device, seed=seed * 100000 + step,
                                draws=draws, sigma=sigma, lr=lr)
            elif name == "backprop":
                res = backprop_step(model, adapter, row, device, lr=lr)
            else:
                raise ValueError("Unknown backend")
            gpu_sync(device)
            history.append({"step": step + 1, **res,
                            "elapsed_seconds": time.monotonic() - start})
        post = loss_on_heldout(model, eval_rows, device)
        gpu_sync(device)
        same_weights = bool(torch.equal(original_weight, adapter.module.weight))
        all_grads_absent = all(p.grad is None for p in model.parameters())
        if not same_weights or not all_grads_absent:
            raise AssertionError("Frozen K2 model weights or gradient state changed")
        changed_b = float(adapter.b.detach().norm().item()) > 0
        if not changed_b:
            raise AssertionError("No LoRA B update occurred")
        return {
            "backend": name, "seed": seed, "lr": lr,
            "draws_per_step": draws if name == "dust" else 0,
            "initial_holdout": pre, "final_holdout": post,
            "heldout_ce_delta": post["mean_token_ce"] - pre["mean_token_ce"],
            "training_history": history,
            "elapsed_seconds": time.monotonic() - start,
            "base_unchanged": same_weights,
            "base_model_grads_absent": all_grads_absent,
            "adapter_b_norm": adapter.b.detach().norm().item(),
            "adapter_a_changed": not torch.equal(adapter.a.detach(), initial_a),
            "backward_calls": 0 if name == "dust" else steps,
            "checkpoint_written": False,
        }
    finally:
        adapter.close()


def run_pilot(*, model_dir: Path, train_file: Path, eval_file: Path,
              device: str = "cuda", expected_sha: str, seed: int = 42,
              train_count: int = 8, eval_count: int = 4,
              train_max_tokens: int = 128, eval_max_tokens: int = 512,
              steps: int = 4, draws: int = 16, sigma: float = 0.05,
              lr: float = 0.1, max_seconds: int = 240) -> dict:
    import torch
    from transformers import AutoTokenizer
    if device not in ("cuda", "cpu"):
        raise ValueError("Unsupported device")
    if not 1 <= steps <= 12 or not 1 <= draws <= 64:
        raise ValueError("Study bounded to <=12 steps and <=64 draws")
    if not 1 <= train_count <= 32 or not 1 <= eval_count <= 12:
        raise ValueError("Study sample counts exceed research limits")
    if not 30 <= max_seconds <= 480:
        raise ValueError("Per-backend wall-clock budget must be 30..480 seconds")
    if not (0 < lr <= 2) or not (0 < sigma <= 0.5):
        raise ValueError("Invalid research learning rate or sigma")
    if train_file.resolve() == eval_file.resolve():
        raise ValueError("Training and heldout source must differ")
    if (model_dir / "config.json").is_file() is False:
        raise FileNotFoundError("Model config missing")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    train, eval_rows, dataset_meta = select_disjoint(
        train_file, eval_file, tokenizer, train_count=train_count,
        eval_count=eval_count, max_tokens=train_max_tokens,
        eval_max_tokens=eval_max_tokens)
    model, source_sha = load_base(model_dir, device=device, expected_sha=expected_sha)
    try:
        baseline = loss_on_heldout(model, eval_rows, device)
        controls = []
        for backend in ("backprop", "dust"):
            controls.append(one_backend(
                model, name=backend, train=train, eval_rows=eval_rows,
                device=device, seed=seed, steps=steps, draws=draws,
                sigma=sigma, lr=lr, max_seconds=max_seconds))
        report = {
            "schema": "auto-finetune.dust-k2-matched-pilot.v1",
            "research_only": True, "production_promotion_authorized": False,
            "pretrained_k2": True, "device": device, "seed": seed,
            "model_weights_sha256": source_sha,
            "model_config_sha256": digest(model_dir / "config.json"),
            "dataset": dataset_meta, "rank": 4, "steps": steps, "draws": draws,
            "learning_rate": lr, "sigma": sigma,
            "matched_train_order": True, "matched_init": True,
            "base_initial_holdout": baseline, "controls": controls,
            "torch": torch.__version__, "hip": torch.version.hip,
            "max_device_memory_allocated": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None),
            "remaining_host_mem_available_bytes": read_mem_available(),
            "raw_user_data_in_report": False, "checkpoint_written": False,
            "warning": "Tiny pilot; prompt-hash disjoint only, semantic leakage and statistical quality unproven",
        }
        return report
    finally:
        del model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-jsonl", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-sha256", default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--train-max-tokens", type=int, default=128)
    parser.add_argument("--eval-max-tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=int, default=240)
    parser.add_argument("--output", type=Path, help="Exclusive create JSON evidence")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing evidence before running")
    result = run_pilot(
        model_dir=args.model_dir, train_file=args.train_jsonl,
        eval_file=args.heldout_jsonl, device=args.device,
        expected_sha=args.expected_sha256, seed=args.seed,
        train_count=args.train_count, eval_count=args.eval_count,
        train_max_tokens=args.train_max_tokens,
        eval_max_tokens=args.eval_max_tokens, steps=args.steps, draws=args.draws,
        sigma=args.sigma, lr=args.lr, max_seconds=args.max_seconds)
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

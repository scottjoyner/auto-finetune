"""Reproducible CPU-only, synthetic-surrogate AdamW/Muon experiment.

This tests benchmark plumbing, not the pretrained LFM3 model or real training.
No private data, external model, checkpoint write, or automatic promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import uuid

from src.lfm3_optimizer_telemetry import observe

SCHEMA = "sensorgraft.lfm3.synthetic-optimizer-ab.v1"
STEPS_MIN, STEPS_MAX = 20, 200
SEEDS_MAX = 8
INPUTS, HIDDEN, OUTPUTS = 16, 32, 8
BASE_SEED = 98765


def _sha_bytes(*chunks):
    h = hashlib.sha256()
    for item in chunks:
        h.update(item)
    return h.hexdigest()


def _digest_tensors(tensors):
    return _sha_bytes(*(t.detach().contiguous().numpy().tobytes() for t in tensors))


def _make_data(torch, seed):
    # Synthetic labels from an independently seeded, frozen teacher.
    g = torch.Generator(device="cpu").manual_seed(BASE_SEED + seed)
    teacher_w = torch.randn(INPUTS, OUTPUTS, generator=g) * 0.2
    teacher_b = torch.randn(OUTPUTS, generator=g) * 0.1
    x_train = torch.randn(128, INPUTS, generator=g)
    x_holdout = torch.randn(64, INPUTS, generator=g)
    # Deliberately different rows, checked by exact byte equality.
    train_rows = {r.contiguous().numpy().tobytes() for r in x_train}
    holdout_rows = {r.contiguous().numpy().tobytes() for r in x_holdout}
    if train_rows & holdout_rows:
        raise RuntimeError("synthetic_train_holdout_overlap")

    def target(x):
        return torch.tanh(x @ teacher_w + teacher_b)

    y_train, y_holdout = target(x_train), target(x_holdout)
    return (
        (x_train, y_train),
        (x_holdout, y_holdout),
        {
            "train_sha256": _digest_tensors([x_train, y_train]),
            "holdout_sha256": _digest_tensors([x_holdout, y_holdout]),
            "heldout_disjoint": True,
            "train_examples": len(x_train),
            "heldout_examples": len(x_holdout),
        },
    )


def _make_model(torch, seed):
    torch.manual_seed(BASE_SEED + 1000 + seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(INPUTS, HIDDEN),
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, OUTPUTS),
    ).to(device="cpu", dtype=torch.float32)
    return model


def _run_arm(torch, *, seed, steps, optimizer_name, lr, adamw_fallback_lr, data, holdout, custody):
    model = _make_model(torch, seed)
    base_digest = _digest_tensors([p for p in model.parameters()])
    all_params = list(model.parameters())
    if optimizer_name == "adamw":
        optimizers = [torch.optim.AdamW(all_params, lr=lr, weight_decay=0)]
    elif optimizer_name == "muon":
        # Only eligible 2D hidden matrices on Muon; 1D biases remain AdamW.
        matrices = [p for p in all_params if p.ndim == 2]
        rest = [p for p in all_params if p.ndim != 2]
        if len(matrices) != 2 or len(rest) != 2:
            raise RuntimeError("unexpected_synthetic_param_groups")
        optimizers = [
            torch.optim.Muon(matrices, lr=lr, weight_decay=0),
            torch.optim.AdamW(rest, lr=adamw_fallback_lr, weight_decay=0),
        ]
    else:
        raise ValueError("optimizer_not_supported")
    x, y = data
    hx, hy = holdout
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        heldout_before = float(criterion(model(hx), hy))
    if not math.isfinite(heldout_before) or heldout_before <= 0:
        raise RuntimeError("invalid_initial_holdout")
    batch = 16
    event_rows = []
    run_id = uuid.uuid4().hex  # unique execution ID; model/data hashes remain deterministic
    for step in range(steps):
        start = time.perf_counter()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        offset = (step * batch) % len(x)
        loss = criterion(model(x[offset : offset + batch]), y[offset : offset + batch])
        loss.backward()
        grad_sq = sum(float(p.grad.detach().float().square().sum()) for p in all_params)
        if not math.isfinite(grad_sq):
            raise RuntimeError("nonfinite_gradient")
        before = [p.detach().clone() for p in all_params]
        for optimizer in optimizers:
            optimizer.step()
        update_sq = sum(
            float((p.detach() - prev).float().square().sum()) for p, prev in zip(all_params, before)
        )
        elapsed = max(time.perf_counter() - start, 1e-9)
        loss_value = float(loss.detach())
        if not all(math.isfinite(v) for v in (loss_value, update_sq, elapsed)):
            raise RuntimeError("nonfinite_training_metric")
        event_rows.append(
            {
                "schema": "sensorgraft.lfm3.optimizer-step.v1",
                "run_id": run_id,
                "step": step,
                "seed": seed,
                "base_sha256": base_digest,
                "dataset_sha256": custody["train_sha256"],
                "optimizer": optimizer_name,
                "learning_rate": lr,
                "train_loss": loss_value,
                "grad_norm": math.sqrt(grad_sq),
                "update_norm": math.sqrt(update_sq),
                "elapsed_s": elapsed,
            }
        )
    with torch.no_grad():
        heldout_after = float(criterion(model(hx), hy))
    if not math.isfinite(heldout_after):
        raise RuntimeError("nonfinite_holdout")
    observation = observe(event_rows)
    if observation["status"] == "blocked":
        raise RuntimeError("telemetry_observer_rejected_run:" + observation["reason"])
    return {
        "seed": seed,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "adamw_fallback_lr": adamw_fallback_lr,
        "steps": steps,
        "base_sha256": base_digest,
        "checkpoint_after_sha256": _digest_tensors(all_params),
        "dataset_sha256": custody["train_sha256"],
        "holdout_sha256": custody["holdout_sha256"],
        "heldout_disjoint": custody["heldout_disjoint"],
        "holdout_loss_before": heldout_before,
        "holdout_loss_after": heldout_after,
        "relative_holdout_change": heldout_after / heldout_before - 1,
        "elapsed_s": sum(r["elapsed_s"] for r in event_rows),
        "events": event_rows,
        "observer": observation,
        "weights_promoted": False,
    }


def benchmark(*, seeds=(1, 2, 3), steps=20, adamw_lr=0.0001, muon_lr=0.001):
    if (
        not isinstance(seeds, (list, tuple))
        or not 1 <= len(seeds) <= SEEDS_MAX
        or any(type(s) is not int or s < 0 or s > 2**32 - 1 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or type(steps) is not int
        or not STEPS_MIN <= steps <= STEPS_MAX
        or type(adamw_lr) is not float
        or not 1e-6 <= adamw_lr <= 2e-4
        or type(muon_lr) is not float
        or not 1e-5 <= muon_lr <= 2e-3
    ):
        raise ValueError("invalid_bounded_benchmark_config")
    import torch

    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch_muon_unavailable")
    torch.set_num_threads(2)
    results = []
    for seed in seeds:
        data, holdout, custody = _make_data(torch, seed)
        arms = [
            _run_arm(
                torch,
                seed=seed,
                steps=steps,
                optimizer_name=name,
                lr=lr,
                adamw_fallback_lr=adamw_lr,
                data=data,
                holdout=holdout,
                custody=custody,
            )
            for name, lr in (("adamw", adamw_lr), ("muon", muon_lr))
        ]
        if (
            arms[0]["base_sha256"] != arms[1]["base_sha256"]
            or arms[0]["dataset_sha256"] != arms[1]["dataset_sha256"]
            or arms[0]["holdout_sha256"] != arms[1]["holdout_sha256"]
            or abs(arms[0]["holdout_loss_before"] - arms[1]["holdout_loss_before"]) > 1e-12
        ):
            raise RuntimeError("unpaired_ab_trial")
        results.extend(arms)
    summary = {
        name: {
            "seed_count": len(seeds),
            "mean_relative_holdout_change": statistics.mean(
                r["relative_holdout_change"] for r in results if r["optimizer"] == name
            ),
            "mean_final_holdout_loss": statistics.mean(
                r["holdout_loss_after"] for r in results if r["optimizer"] == name
            ),
        }
        for name in ("adamw", "muon")
    }
    return {
        "schema": SCHEMA,
        "status": "synthetic_surrogate_only",
        "model": "tiny_cpu_mlp_not_lfm3",
        "real_private_traces_used": False,
        "dataset_type": "synthetic_teacher_generated",
        "production_ready": False,
        "weight_promotion_permitted": False,
        "seeds": list(seeds),
        "steps_per_arm": steps,
        "arms": results,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--adamw-lr", type=float, default=0.0001)
    parser.add_argument("--muon-lr", type=float, default=0.001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = benchmark(
        seeds=args.seeds,
        steps=args.steps,
        adamw_lr=args.adamw_lr,
        muon_lr=args.muon_lr,
    )
    with open(args.output, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "output": args.output,
                "weight_promotion_permitted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
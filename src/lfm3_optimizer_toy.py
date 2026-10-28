"""Isolated CPU-only AdamW vs Muon hybrid on an original tiny toy network.

No LFM3 weights are loaded or tuned. This is optimizer-mechanics research:
fresh initialization per arm, fixed fabricated train/dev data and step budget,
full in-memory cleanup, no checkpoint writing or production triggers.
"""
from __future__ import annotations

import hashlib
import json
import math
import resource
import time

from src.lfm3_optimizer_guard import (
    CHECKPOINT, OptimizerResearchRefused, compare_dev_only,
    fingerprint, observer_view, validate_observation,
)

UNSAFE = ("embed", "lm_head", "output", "readout", "norm", "bias",
          "conv", "lora", "classifier", "router", "gate")
STEPS = 24
SEED = 71


def classify_for_muon(named_parameters, reviewed_hidden_weights: set[str]):
    """Only explicitly reviewed, genuinely 2-D hidden matrices may use Muon.

    Shape alone is not permission: token embeddings and LM heads also have
    matrix-shaped weights. LoRA adapters and LFM conv/gates default to AdamW.
    """
    if type(reviewed_hidden_weights) is not set or not reviewed_hidden_weights:
        raise OptimizerResearchRefused("explicit reviewed hidden weight allowlist required")
    muon = []
    adamw = []
    seen_names = set()
    seen_tensors = set()
    reviewed_seen = set()
    for name, param in named_parameters:
        if (type(name) is not str or name in seen_names
                or id(param) in seen_tensors):
            raise OptimizerResearchRefused("optimizer parameter alias or duplicate")
        seen_names.add(name)
        seen_tensors.add(id(param))
        if not getattr(param, "requires_grad", False):
            continue
        if name in reviewed_hidden_weights:
            pieces = name.lower().split(".")
            if (not name.endswith(".weight")
                    or any(block in piece for piece in pieces for block in UNSAFE)
                    or getattr(param, "ndim", None) != 2
                    or min(param.shape) <= 1):
                raise OptimizerResearchRefused("Muon requires reviewed 2-D hidden matrix")
            muon.append(param)
            reviewed_seen.add(name)
        else:
            adamw.append(param)
    if reviewed_seen != reviewed_hidden_weights or not muon:
        raise OptimizerResearchRefused("missing or frozen reviewed Muon matrices")
    return muon, adamw


def optimizers_for_toy(model, optimizer: str):
    import torch
    names = list(model.named_parameters())
    muon, adamw = classify_for_muon(names, {"hidden.weight"})
    if optimizer == "adamw":
        return [torch.optim.AdamW(muon + adamw, lr=0.001, weight_decay=0.01)]
    if optimizer != "muon_hybrid":
        raise OptimizerResearchRefused("unsupported optimizer")
    if not hasattr(torch.optim, "Muon"):
        raise OptimizerResearchRefused("native torch.optim.Muon unavailable")
    out = [torch.optim.Muon(
        muon, lr=0.001, momentum=0.95, weight_decay=0.01,
        ns_steps=5, adjust_lr_fn="original",
    )]
    if adamw:
        out.append(torch.optim.AdamW(
            adamw, lr=0.001, weight_decay=0.01
        ))
    return out


def run_toy_arm(optimizer: str, *, steps: int = STEPS) -> dict:
    if optimizer not in ("adamw", "muon_hybrid") or not 1 <= steps <= 64:
        raise OptimizerResearchRefused("toy training optimizer or budget refused")
    import torch
    from torch import nn
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train_x = torch.randn((96, 12), generator=generator)
    dev_x = torch.randn((48, 12), generator=generator)
    truth = torch.randn((12, 4), generator=generator)
    train_y = torch.tanh(train_x @ truth)
    dev_y = torch.tanh(dev_x @ truth)
    # Model params are reset from exactly the same seed for BOTH optimizer arms.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = nn.Sequential(
            __import__("collections").OrderedDict([
                ("hidden", nn.Linear(12, 24)),
                ("act", nn.Tanh()),
                ("head", nn.Linear(24, 4)),
            ])
        ).cpu().float()
    model.train()
    opts = optimizers_for_toy(model, optimizer)
    grad_peak = 0.0
    nonfinite = 0
    elapsed = []
    start_loss = None
    for _ in range(steps):
        started = time.perf_counter()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(train_x), train_y)
        if start_loss is None:
            start_loss = float(loss.detach())
        if not bool(torch.isfinite(loss)):
            nonfinite += 1
            break
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not bool(torch.isfinite(grad_norm)):
            nonfinite += 1
            break
        grad_peak = max(grad_peak, float(grad_norm))
        for o in opts:
            o.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            nonfinite += 1
            break
        elapsed.append((time.perf_counter() - started) * 1000)
    if not elapsed:
        raise OptimizerResearchRefused("toy optimizer never completed a step")
    model.eval()
    with torch.no_grad():
        train_end = float(nn.functional.mse_loss(model(train_x), train_y))
        dev_loss = float(nn.functional.mse_loss(model(dev_x), dev_y))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    train_label = hashlib.sha256(b"sg-toy-train-fabricated-v1").hexdigest()
    dev_label = hashlib.sha256(b"sg-toy-dev-fabricated-v1").hexdigest()
    result = {
        "schema": "sg-lfm3-optimizer-observation.v1",
        "source": "fabricated-synthetic-lab-only",
        "model_scope": "toy-proxy-not-lfm3-finetuning",
        "reference_lfm3_checkpoint_sha256": CHECKPOINT,
        "run_id": "lfm3-optim-toy-20261027",
        "train_session_sha256": train_label, "dev_session_sha256": dev_label,
        "seed": SEED, "steps": steps, "optimizer": optimizer,
        "learning_rate": 0.001, "weight_decay": 0.01,
        "momentum": 0.95, "ns_steps": 5 if optimizer == "muon_hybrid" else 0,
        "train_loss_start": start_loss, "train_loss_end": train_end,
        "dev_loss": dev_loss, "max_grad_norm": grad_peak,
        "step_mean_ms": sum(elapsed) / len(elapsed),
        "peak_memory_mb": peak, "nonfinite_updates": nonfinite,
        "actual_lfm3_weights_trained": False,
        "model_weights_written": False, "production_ready": False,
    }
    validate_observation(result)
    return result


def run_pair() -> dict:
    adamw = run_toy_arm("adamw")
    muon = run_toy_arm("muon_hybrid")
    comparison = compare_dev_only(adamw, muon)
    return {
        "schema": "sg-lfm3-optimizer-toy-paired.v1",
        "fabricated_only": True,
        "trained_real_lfm3": False,
        "optimization_target": "randomly_initialized_small_dense_toy",
        "torch_optim_muon_executed": True,
        "torch_optim_adamw_executed": True,
        "fresh_init_per_arm": True,
        "baseline": adamw,
        "candidate": muon,
        "comparison": comparison,
        "observer_input": observer_view(adamw),
        "may_promote": False,
        "model_checkpoint_written": False,
        "production_ready": False,
    }


def main():
    print(json.dumps(run_pair(), sort_keys=True))


if __name__ == "__main__":
    main()

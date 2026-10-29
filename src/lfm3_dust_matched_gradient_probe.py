"""Pinned LFM3-1.7B LoRA-B-only gradient controls for Dust comparisons.

This deliberately restricts AdamW and experimental factor-wise Muon to the
SAME last-two v_proj LoRA-B matrices (2,048 scalars) as Dust, while leaving all
other Q/V LoRA factors and pretrained weights frozen. It is a fair *parameter
scope* control, NOT an equal-forward or equal-walltime experiment.

Offline, CPU-only opt-in research. No adapters exported, no checkpoints
written, no model selection or production routing changed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path

from src.lfm3_lora_text_task_probe import _completion_loss, _digest, _evaluate, dataset
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA = "sensorgraft.lfm3.matched-lora-b-gradient-controls.v1"
STEPS = 3
TRUST_RADIUS = 0.15
ADAMW_LR = 0.0001
MUON_LR = 0.00016
EXPECTED_PARAMS = 2048


def validate(*, seed, optimizer, steps, adamw_lr=ADAMW_LR, muon_lr=MUON_LR):
    if (type(seed) is not int or seed not in (1, 2, 3)
            or type(steps) is not int or steps != STEPS
            or type(optimizer) is not str or optimizer not in ("adamw", "muon")
            or type(adamw_lr) not in (int, float) or not math.isfinite(adamw_lr)
            or type(muon_lr) not in (int, float) or not math.isfinite(muon_lr)
            or adamw_lr != ADAMW_LR or muon_lr != MUON_LR):
        raise ValueError("nonpreregistered_lfm3_matched_parameter_config")


def select_b_only(named):
    candidate = [(name, p) for name, p in named
                 if ".self_attn.v_proj." in name and ".lora_B." in name]
    chosen = sorted(
        candidate,
        key=lambda row: int(row[0].split(".layers.")[1].split(".")[0]),
    )[-2:]
    if (len(chosen) != 2
            or sum(p.numel() for _, p in chosen) != EXPECTED_PARAMS
            or not all(p.ndim == 2 for _, p in chosen)
            or not all(n.endswith(".self_attn.v_proj.lora_B.default.weight")
                       for n, _ in chosen)):
        raise RuntimeError("unverified_lfm3_matching_adapter_scope")
    return chosen


def run(*, seed=1, optimizer="adamw", steps=STEPS,
        model_path=MODEL_PATH, adamw_lr=ADAMW_LR, muon_lr=MUON_LR):
    validate(seed=seed, optimizer=optimizer, steps=steps,
             adamw_lr=adamw_lr, muon_lr=muon_lr)
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3

    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    model, named = _load(torch, seed, str(model_path))
    chosen = select_b_only(named)
    model.requires_grad_(False)
    for _, p in chosen:
        p.requires_grad_(True)
    params = [p for _, p in chosen]
    if (sum(p.numel() for p in model.parameters() if p.requires_grad)
            != EXPECTED_PARAMS):
        raise RuntimeError("unintended_parameter_admission")
    originals = [p.detach().clone() for p in params]
    initial_hash = _sha(*params)
    backbone = next(p for n, p in model.named_parameters()
                    if n.endswith("model.layers.2.self_attn.q_proj.base_layer.weight"))
    backbone_sample = backbone.detach().reshape(-1)[:256].clone()
    parts = dataset(seed)
    manifests = {name: _digest(items) for name, items in parts.items()}
    model.eval()
    before = _evaluate(torch, model, tokenizer, parts["selection"])
    lr = adamw_lr if optimizer == "adamw" else muon_lr
    if optimizer == "muon" and not hasattr(torch.optim, "Muon"):
        raise RuntimeError("experimental_torch_muon_unavailable")
    optim = (torch.optim.AdamW(params, lr=lr, weight_decay=0)
             if optimizer == "adamw" else
             torch.optim.Muon(params, lr=lr, weight_decay=0))
    rows = []
    t0 = time.perf_counter()
    for step in range(steps):
        model.train()
        optim.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = _completion_loss(torch, model, tokenizer, parts["train"][step])
        loss.backward()
        raw_grad = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0))
        if not math.isfinite(raw_grad) or raw_grad > 1e5:
            raise RuntimeError("unsafe_or_nonfinite_gradient")
        previous = [p.detach().clone() for p in params]
        optim.step()
        update_sq = sum(float((p.detach() - b).float().square().sum())
                        for p, b in zip(params, previous, strict=True))
        drift_sq = sum(float((p.detach() - b).float().square().sum())
                       for p, b in zip(params, originals, strict=True))
        if not all(math.isfinite(x) for x in (update_sq, drift_sq)):
            raise RuntimeError("nonfinite_update_or_adapter_drift")
        drift = math.sqrt(drift_sq)
        if drift > TRUST_RADIUS:
            raise RuntimeError("matched_parameter_trust_radius_exceeded")
        rows.append({
            "step": step, "optimizer": optimizer,
            "learning_rate": lr, "train_loss": float(loss.detach()),
            "raw_grad_norm": raw_grad, "clipped_grad_max_norm": 1.0,
            "update_norm": math.sqrt(update_sq),
            "cumulative_lora_b_update_norm": drift,
            "elapsed_s": max(time.perf_counter() - started, 1e-9),
            "actual_forward_backwards": 1,
        })
    model.eval()
    after = _evaluate(torch, model, tokenizer, parts["selection"])
    if not torch.equal(backbone_sample, backbone.detach().reshape(-1)[:256]):
        raise RuntimeError("frozen_pretrained_backbone_mutated")
    if any(p.requires_grad for n, p in model.named_parameters()
           if n not in {k for k, _ in chosen}):
        raise RuntimeError("nonapproved_trainable_parameter")
    return {
        "schema": SCHEMA, "status": "research_parameter_matched_gradient_control",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE_SHA256,
        "seed": seed, "optimizer": optimizer, "steps": steps,
        "run_id": uuid.uuid4().hex, "learning_rate": lr,
        "parameter_scope": "last_two_attention_vproj_lora_B_only",
        "selected_adapter_names": [name for name, _ in chosen],
        "selected_parameter_count": EXPECTED_PARAMS,
        "initial_lora_b_sha256": initial_hash,
        "final_lora_b_sha256": _sha(*params),
        "data_manifest": manifests,
        "selection_before": before, "selection_after": after,
        "records": rows, "actual_backwards": steps,
        "training_forwards": steps,
        "selection_score_forwards": 36,
        "no_checkpoints_written": True,
        "final_test_evaluated": False,
        "equal_parameter_scope_with_dust": True,
        "equal_total_forward_or_walltime_budget_with_dust": False,
        "comparable_to_full_qv_lora_optimization": False,
        "experimental_muon_on_lora_factors": optimizer == "muon",
        "elapsed_training_s": max(time.perf_counter() - t0, 1e-9),
        "source_independently_authenticated": False,
        "trial_authorized": False,
        "weight_promotion_permitted": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--allow-matched-lora-b-gradient-control", action="store_true",
                   required=True)
    p.add_argument("--allow-experimental-muon-lora", action="store_true")
    p.add_argument("--optimizer", choices=("adamw", "muon"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.optimizer == "muon" and not a.allow_experimental_muon_lora:
        p.error("Muon requires explicit experimental opt-in")
    result = run(seed=a.seed, optimizer=a.optimizer, steps=a.steps,
                 model_path=a.model_path)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({"status": result["status"], "seed": a.seed,
                      "optimizer": a.optimizer,
                      "selection_before": result["selection_before"]["correct"],
                      "selection_after": result["selection_after"]["correct"],
                      "output": str(a.output),
                      "weight_promotion_permitted": False}))
if __name__ == "__main__":
    main()

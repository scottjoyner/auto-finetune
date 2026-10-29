"""Third optimizer research arm: Dust-style forward-only LFM3 LoRA-B trial.

Uses a causal online direction-attention head to RANK candidate perturbation
directions based solely on previous train-only scalar probe outcomes. All
selected proposals are independently tested via antithetic + and - forward
passes BEFORE a bounded LoRA-B parameter change is considered. Neither the
backbone nor Q/V LoRA-A are trained; this is not an apples-to-apples comparison
with previous full-adapter AdamW/Muon work.

Strict offline CPU mode, fresh pinned pretrained load, no backward/autograd,
no optimizer object, no checkpoint saves, no final-test scoring, no promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from pathlib import Path

from src.lfm3_dust_direction_attention import DIM, DirectionAttentionHead
from src.lfm3_dust_activation_head import ActivationDirectionHead, prompt_only_activation_context
from src.lfm3_dust_online_calibration import CalibratedActivationHead
from src.lfm3_lora_text_task_probe import _completion_loss, _digest, _evaluate, dataset
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA = "sensorgraft.lfm3.dust-forward-text-pilot.v1"
SIGMA = 0.05
MAX_RADIUS = 0.15
CANDIDATES = 4
DRAWS = 2
MAX_STEPS = 4


def validate(*, seed, steps, draws, policy):
    if (type(seed) is not int or seed not in (1, 2, 3)
            or type(steps) is not int or not 1 <= steps <= MAX_STEPS
            or type(draws) is not int or draws != DRAWS
            or policy not in ("attended", "random", "activation_attended", "activation_calibrated")):
        raise ValueError("out_of_bounds_zeroth_order_training_plan")


def _unit(x):
    n = math.sqrt(sum(v * v for v in x))
    if not math.isfinite(n) or n < 1e-8:
        raise ValueError("degenerate_direction")
    return tuple(v / n for v in x)


def _basis(torch, params, *, seed):
    generator = torch.Generator(device="cpu").manual_seed(730021 + seed)
    outputs = []
    for _ in range(DIM):
        blocks = [torch.randn(p.shape, generator=generator,
                              dtype=torch.float32, device="cpu") for p in params]
        total = math.sqrt(sum(float(b.square().sum()) for b in blocks))
        outputs.append([b / total for b in blocks])
    return outputs


def _direction(torch, basis, features):
    # Features are pre-probe and span only the fixed low-dimensional basis.
    blocks = [
        sum((coeff * basis[j][i] for j, coeff in enumerate(features)))
        for i in range(len(basis[0]))
    ]
    norm = math.sqrt(sum(float(b.square().sum()) for b in blocks))
    if not 0 < norm < 100:
        raise ValueError("nonfinite_basis_direction")
    return [b / norm for b in blocks]


def _loss(torch, model, tokenizer, item):
    with torch.inference_mode():
        loss = float(_completion_loss(torch, model, tokenizer, item))
    if not math.isfinite(loss) or not 0 < loss < 1000:
        raise RuntimeError("nonfinite_forward_only_loss")
    return loss


def _apply(torch, params, blocks, multiplier):
    with torch.no_grad():
        for p, d in zip(params, blocks, strict=True):
            p.add_(d, alpha=multiplier)


def _radius(params, originals):
    return math.sqrt(sum(float((p.detach() - b).float().square().sum())
                         for p, b in zip(params, originals, strict=True)))


def run(*, seed=1, steps=2, draws=DRAWS, policy="attended",
        model_path=MODEL_PATH):
    validate(seed=seed, steps=steps, draws=draws, policy=policy)
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3

    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    model, named = _load(torch, seed, str(model_path))
    # The experiment is intentionally a restricted low-dimensional B-only
    # experiment; no use of .backward(), torch.optim, or gradient recording.
    model.requires_grad_(False)
    model.eval()
    eligible = [(name, p) for name, p in named
                if ".self_attn.v_proj." in name and ".lora_B." in name]
    chosen = sorted(eligible, key=lambda pair: int(pair[0].split(".layers.")[1].split(".")[0]))[-2:]
    if len(chosen) != 2 or any(p.dtype != torch.float32 for _, p in chosen):
        raise RuntimeError("unexpected_lfm3_vproj_adapter_selection")
    params = [p for _, p in chosen]
    originals = [p.detach().clone() for p in params]
    initial_hash = _sha(*params)
    if any(x.requires_grad for x in model.parameters()):
        raise RuntimeError("autograd_trainables_not_disabled")
    # Actual gradients must remain wholly absent.
    if any(x.grad is not None for x in model.parameters()):
        raise RuntimeError("model_gradient_residue")

    splits = dataset(seed)
    manifests = {section: _digest(rows) for section, rows in splits.items()}
    before = _evaluate(torch, model, tokenizer, splits["selection"])
    head = (CalibratedActivationHead() if policy == "activation_calibrated"
            else ActivationDirectionHead() if policy == "activation_attended"
            else DirectionAttentionHead())
    basis = _basis(torch, params, seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(851003 + seed)
    rows = []
    t0 = time.perf_counter()
    for step in range(steps):
        item = splits["train"][step]
        # Context comes only from prompt tokens and is obtained BEFORE
        # the current candidate has any perturbation outcome.
        context = (prompt_only_activation_context(torch, model, tokenizer, item["prompt"])
                   if policy in ("activation_attended", "activation_calibrated") else None)
        clean = _loss(torch, model, tokenizer, item)
        proposals = [_unit([float(x) for x in torch.randn(
            DIM, generator=generator, dtype=torch.float32)])
            for _ in range(CANDIDATES)]
        preprobe_order = (head.rank(proposals, context, episode=step)
                          if context is not None else head.rank(proposals, episode=step))
        selected = (preprobe_order[:draws] if policy in ("attended", "activation_attended", "activation_calibrated")
                    else list(range(draws)))
        predictions = [(head.predict(features, context, episode=step)
                        if context is not None else head.predict(features, episode=step))
                       for features in proposals]
        score_pre = [entry["predicted_gain"] for entry in predictions]
        gate_pre = (list(head.weights) if policy == "activation_calibrated"
                    else None)
        calibration_pre = (head.calibration_events if gate_pre is not None else None)
        best = (clean, None, 0.0)
        gains = []
        for candidate in selected:
            features = proposals[candidate]
            blocks = _direction(torch, basis, features)
            # Exactly restore original B after each antithetic probe, even
            # when a forward pass errors; never write to base checkpoint.
            _apply(torch, params, blocks, +SIGMA)
            try:
                plus = _loss(torch, model, tokenizer, item)
            finally:
                _apply(torch, params, blocks, -SIGMA)
            _apply(torch, params, blocks, -SIGMA)
            try:
                minus = _loss(torch, model, tokenizer, item)
            finally:
                _apply(torch, params, blocks, +SIGMA)
            if not all(math.isfinite(x) for x in (plus, minus)):
                raise RuntimeError("invalid_antithetic_pair")
            if context is not None:
                head.observe(features, context, clean - plus, episode=step, measured=True)
                head.observe(tuple(-v for v in features), context,
                             clean - minus, episode=step, measured=True)
            else:
                head.observe(features, clean - plus, episode=step)
                head.observe(tuple(-v for v in features), clean - minus, episode=step)
            gains.append({"candidate": candidate, "predicted_gain": score_pre[candidate],
                          "plus_gain": clean - plus, "minus_gain": clean - minus})
            if plus + 1e-6 < best[0]:
                best = (plus, blocks, +SIGMA)
            if minus + 1e-6 < best[0]:
                best = (minus, blocks, -SIGMA)
        after = clean
        accepted = False
        if best[1] is not None:
            _apply(torch, params, best[1], best[2])
            # Refuse step if accumulated perturbation exceeds trust radius.
            if _radius(params, originals) <= MAX_RADIUS:
                after = _loss(torch, model, tokenizer, item)
                if after <= clean + 1e-5:
                    accepted = True
            if not accepted:
                _apply(torch, params, best[1], -best[2])
                after = clean
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("unexpected_backward_gradient")
        rows.append({
            "step": step, "train_loss_before": clean, "train_loss_after": after,
            "selected_candidates": selected, "preprobe_order": preprobe_order,
            "prior_memory_entries": head.size - 2 * len(selected),
            "antithetic_probe_results": gains, "step_accepted": accepted,
            "cumulative_lora_b_update_norm": _radius(params, originals),
            "actual_backwards": 0, "actual_antithetic_forwards": 2 * draws,
            **({"prompt_context_sha256": hashlib.sha256(json.dumps(
                   context, separators=(",", ":"), allow_nan=False
               ).encode("utf-8")).hexdigest(),
               "attention_head_disagreement": [entry["head_disagreement"]
                                               for entry in predictions],
               "prompt_only_extra_forwards": 1}
               if context is not None else {}),
            **({"calibration_weights_before_probes": gate_pre,
               "calibration_events_before_probes": calibration_pre,
               "calibration_events_after_probes": head.calibration_events,
               "calibration_weights_after_probes": list(head.weights)}
               if gate_pre is not None else {}),
        })
    after = _evaluate(torch, model, tokenizer, splits["selection"])
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("unexpected_final_gradients")
    if _radius(params, originals) > MAX_RADIUS:
        raise RuntimeError("trust_region_exceeded")
    result = {
        "schema": SCHEMA, "status": "research_forward_only_text_diagnostic",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE_SHA256,
        "seed": seed, "steps": steps, "policy": policy,
        "direction_head": ("causal_prompt_activation_two_head_train_online_calibrated"
                           if policy == "activation_calibrated" else
                           "causal_prompt_activation_two_head_training_probes_only"
                           if policy == "activation_attended" else
                           "causal_online_kernel_attention_training_probes_only"),
        "adapter_policy": "last_two_attention_vproj_lora_B_only",
        "selected_adapter_names": [name for name, _ in chosen],
        "selected_parameter_count": sum(p.numel() for p in params),
        "initial_lora_b_sha256": initial_hash,
        "final_lora_b_sha256": _sha(*params),
        "data_manifest": manifests,
        "selection_before": before, "selection_after": after,
        "records": rows, "gradient_backwards": 0,
        "optimizer_object_used": False,
        "forward_only": True,
        "total_antithetic_forwards": sum(x["actual_antithetic_forwards"] for x in rows),
        **({"extra_prompt_only_forwards": steps, "raw_activations_persisted": False}
           if policy in ("activation_attended", "activation_calibrated") else {}),
        "total_accepted_steps": sum(x["step_accepted"] for x in rows),
        "elapsed_s": time.perf_counter() - t0,
        "final_test_evaluated": False, "real_task_generalization_established": False,
        "comparable_to_full_lora_adamw_muon": False,
        "model_weight_checkpoint_written": False,
        "trial_authorized": False, "weight_promotion_permitted": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-only", action="store_true", required=True)
    parser.add_argument("--allow-backprop-free-lora-probe", action="store_true", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--policy", choices=("attended", "random", "activation_attended", "activation_calibrated"), default="attended")
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(seed=args.seed, steps=args.steps, draws=args.draws,
                 policy=args.policy, model_path=args.model_path)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "status": result["status"], "seed": result["seed"], "policy": result["policy"],
        "selection_before": result["selection_before"]["correct"],
        "selection_after": result["selection_after"]["correct"],
        "accepted_steps": result["total_accepted_steps"],
        "backwards": 0, "output": str(args.output),
        "weight_promotion_permitted": False
    }, sort_keys=True))


if __name__ == "__main__":
    main()

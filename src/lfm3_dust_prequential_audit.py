"""Prospective, all-candidate LFM3 Dust forecast audit with NO training updates.

The task is not optimizer selection: it asks whether past training-only scalar
forward measurements predict the gain of entirely new candidate perturbations.
Every candidate is independently measured, not only predictor-selected ones,
so ranking metrics are not confounded by selective probing.

The pinned real pretrained model is frozen. Temporary antithetic LoRA-B
perturbations are reset by exact copy in finally (no cumulative roundoff).
No model, adapter, or calibrated head coefficients are saved, no test split is
opened, no gradients/backprop/optimizer objects or deployed services touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import uuid
from pathlib import Path

from src.lfm3_dust_activation_head import (
    ActivationDirectionHead, prompt_only_activation_context,
)
from src.lfm3_dust_direction_attention import DIM, DirectionAttentionHead
from src.lfm3_dust_online_calibration import CalibratedActivationHead
from src.lfm3_dust_predictive_probe import _basis, _direction, _loss, _unit
from src.lfm3_lora_text_task_probe import dataset, _digest
from src.lfm3_real_lora_probe import BASE_SHA256, MODEL_PATH, _load, _preflight, _sha

SCHEMA = "sensorgraft.lfm3.dust-prequential-all-candidates.v1"
EPISODES = 8
CANDIDATES = 4
SIGMA = .05
POLICIES = ("zero", "historical_mean", "direction", "activation", "calibrated")


def validate(*, seed, episodes, candidates, sigma):
    if (type(seed) is not int or seed not in (1, 2, 3)
            or type(episodes) is not int or episodes != EPISODES
            or type(candidates) is not int or candidates != CANDIDATES
            or type(sigma) not in (int, float) or sigma != SIGMA):
        raise ValueError("unpreregistered_prequential_experiment")


def _metrics(episodes):
    """No information from the current episode may enter its forecasts."""
    evaluation = [r for r in episodes if r["episode"] > 0]
    if len(evaluation) != EPISODES - 1:
        raise ValueError("incomplete_prequential_period")
    result = {}
    for policy in POLICIES:
        sq, absolute, selected, oracle, observed = [], [], [], [], []
        for row in evaluation:
            values = row["candidates"]
            estimates = [c["predictions"][policy] for c in values]
            gains = [c["measured_gain"] for c in values]
            if not all(math.isfinite(x) for x in estimates + gains):
                raise ValueError("nonfinite_prediction_or_gain")
            sq.extend((p - g) ** 2 for p, g in zip(estimates, gains, strict=True))
            absolute.extend(abs(p - g) for p, g in zip(estimates, gains, strict=True))
            chosen = max(range(len(values)), key=lambda i: (estimates[i], -i))
            selected.append(gains[chosen])
            oracle.append(max(gains))
            observed.extend(gains)
        result[policy] = {
            "cases": len(sq),
            "mse": statistics.mean(sq),
            "mae": statistics.mean(absolute),
            "mean_gain_if_top1_probed": statistics.mean(selected),
            "mean_oracle_best_gain": statistics.mean(oracle),
            "mean_top1_regret": statistics.mean(x-y for x,y in zip(oracle,selected,strict=True)),
        }
    return result


def run(*, seed=1, episodes=EPISODES, candidates=CANDIDATES,
        sigma=SIGMA, model_path=MODEL_PATH):
    validate(seed=seed, episodes=episodes, candidates=candidates, sigma=sigma)
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3
    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    model, named = _load(torch, seed, str(model_path))
    model.requires_grad_(False)
    model.eval()
    eligible = [(name, p) for name, p in named
                if ".self_attn.v_proj." in name and ".lora_B." in name]
    chosen = sorted(eligible, key=lambda x: int(x[0].split(".layers.")[1].split(".")[0]))[-2:]
    if (len(chosen) != 2
            or sum(p.numel() for _, p in chosen) != 2048
            or any(p.requires_grad for p in model.parameters())
            or any(p.grad is not None for p in model.parameters())):
        raise RuntimeError("invalid_frozen_lfm3_adapter_scope")
    params = [p for _, p in chosen]
    saved = [p.detach().clone() for p in params]
    original_sha = _sha(*params)
    basis = _basis(torch, params, seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(851003 + seed)
    direction = DirectionAttentionHead()
    activation = ActivationDirectionHead()
    calibrated = CalibratedActivationHead()
    parts = dataset(seed)
    manifests = {name: _digest(rows) for name, rows in parts.items()}
    prior_gains = []
    episode_rows = []
    started = time.perf_counter()
    for ep in range(episodes):
        item = parts["train"][ep]
        # The additional context forward is prompt-only, and precedes all
        # current-episode outcome measurements.
        context = prompt_only_activation_context(torch, model, tokenizer, item["prompt"])
        clean = _loss(torch, model, tokenizer, item)
        proposals = [_unit([float(x) for x in torch.randn(
            DIM, generator=generator, dtype=torch.float32)]) for _ in range(candidates)]
        entries = []
        for idx, features in enumerate(proposals):
            for sign in (+1, -1):
                signed = tuple(sign * v for v in features)
                d = direction.predict(signed, episode=ep)
                a = activation.predict(signed, context, episode=ep)
                c = calibrated.predict(signed, context, episode=ep)
                prior_mean = statistics.mean(prior_gains) if prior_gains else 0.
                entries.append({
                    "candidate": idx, "sign": sign,
                    "direction_coefficients": list(signed),
                    "predictions": {
                        "zero": 0.0, "historical_mean": prior_mean,
                        "direction": d["predicted_gain"],
                        "activation": a["predicted_gain"],
                        "calibrated": c["predicted_gain"],
                    },
                    "direction_evidence": d["evidence"],
                    "activation_evidence": a["evidence"],
                    "calibrated_evidence": c["evidence"],
                    "head_weights_preprobe": list(c["head_weights"]),
                })
        # Frozen forecasts for ALL 2*CANDIDATES directions: no selective
        # winner's curse and no same-episode outcome-to-prediction leakage.
        for entry in entries:
            features = tuple(entry["direction_coefficients"])
            blocks = _direction(torch, basis, features)
            with torch.no_grad():
                for p, original, delta in zip(params, saved, blocks, strict=True):
                    p.copy_(original + sigma * delta)
            try:
                perturbed = _loss(torch, model, tokenizer, item)
            finally:
                with torch.no_grad():
                    for p, original in zip(params, saved, strict=True):
                        p.copy_(original)
            if _sha(*params) != original_sha:
                raise RuntimeError("antithetic_probe_not_exactly_restored")
            entry["measured_gain"] = clean - perturbed
            if not math.isfinite(entry["measured_gain"]) or abs(entry["measured_gain"]) > 100:
                raise RuntimeError("unbounded_prequential_gain")
        # Only after ALL candidate forecasts and measurements are recorded
        # may their labels update the heads for later episodes.
        for entry in entries:
            direction.observe(entry["direction_coefficients"], entry["measured_gain"],
                              episode=ep, split="train", probed=True)
            activation.observe(entry["direction_coefficients"], context,
                               entry["measured_gain"], episode=ep,
                               split="train", measured=True)
            calibrated.observe(entry["direction_coefficients"], context,
                               entry["measured_gain"], episode=ep,
                               split="train", measured=True)
            prior_gains.append(entry["measured_gain"])
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("unexpected_training_gradient")
        payload = {
            "episode": ep,
            "train_example_sha256": hashlib.sha256(
                json.dumps({k:v for k,v in item.items() if k!="input_key"},
                           sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "context_sha256": hashlib.sha256(
                json.dumps(context, separators=(",", ":")).encode()).hexdigest(),
            "train_clean_loss": clean,
            "calibrated_weights_after_feedback": list(calibrated.weights),
            "calibration_event_count_after": calibrated.calibration_events,
            "candidates": entries,
            "perturbed_forward_count": 2 * candidates,
            "prompt_only_forward_count": 1,
            "training_backward_count": 0,
        }
        episode_rows.append(payload)
        print(json.dumps({
            "seed": seed, "episode": ep,
            "gate_direction_weight": round(calibrated.weights[0], 5),
            "measured_probes": len(entries),
        }), flush=True)
    if (_sha(*params) != original_sha
            or any(not torch.equal(p.detach(), original)
                   for p, original in zip(params, saved, strict=True))):
        raise RuntimeError("frozen_adapter_modified_after_observation")
    metrics = _metrics(episode_rows)
    return {
        "schema": SCHEMA, "status": "real_lfm3_prequential_prediction_only",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE_SHA256,
        "seed": seed, "episodes": episodes,
        "candidates_per_episode": candidates,
        "sigma": sigma, "model_run_id": uuid.uuid4().hex,
        "parameter_scope": "last_two_attention_vproj_lora_B_only",
        "observed_parameter_count": sum(p.numel() for p in params),
        "initial_adapter_sha256": original_sha,
        "final_adapter_sha256": _sha(*params),
        "data_manifest": manifests,
        "metrics_exclude_cold_start": True,
        "metrics": metrics, "episode_receipts": episode_rows,
        "total_perturbed_forwards": 2 * candidates * episodes,
        "total_context_forwards": episodes,
        "total_backward_calls": 0,
        "model_adapter_update_count": 0,
        "independent_test_evaluated": False,
        "selection_partition_evaluated": False,
        "private_traces_used": False,
        "source_independently_authenticated": False,
        "equal_compute_to_adamw_muon": False,
        "model_quality_improvement_proven": False,
        "calibration_useful_for_generalization_proven": False,
        "elapsed_s": max(time.perf_counter() - started, 1e-9),
        "weight_promotion_permitted": False, "trial_authorized": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--allow-forward-only-all-candidate-audit", action="store_true", required=True)
    p.add_argument("--seed", type=int, choices=(1,2,3), required=True)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--candidates", type=int, default=CANDIDATES)
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(seed=args.seed, episodes=args.episodes,
                 candidates=args.candidates, sigma=args.sigma,
                 model_path=args.model_path)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "seed": args.seed, "status": result["status"],
        "metric_policies": list(result["metrics"]),
        "model_adapter_update_count": 0,
        "independent_test_evaluated": False,
        "weight_promotion_permitted": False,
        "output": str(args.output),
    }, allow_nan=False))


if __name__ == "__main__":
    main()

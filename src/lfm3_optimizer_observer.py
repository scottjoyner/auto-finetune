"""Read-only LFM3 optimizer observer; never modifies weights or training jobs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from src.lfm3_real_lora_review import BASE, review

SCHEMA = "sensorgraft.lfm3.optimizer-observer.v1"
LIMITS = {"adamw": (1e-6, 2e-4), "muon": (1e-5, 2e-3)}


def blocked(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "trial_authorized": False, "weight_promotion_permitted": False}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _summarize(arm):
    events = arm["events"]
    grad = [e["grad_norm"] for e in events]
    update = [e["update_norm"] for e in events]
    elapsed = [e["elapsed_s"] for e in events]
    median_grad, median_update = statistics.median(grad), statistics.median(update)
    anomaly = (not all(_finite(v) and 0 <= v <= 1e6 for v in grad + update)
               or not all(_finite(v) and 0 < v <= 3600 for v in elapsed)
               or max(grad) > 50 * max(median_grad, 1e-12)
               or max(update) > 50 * max(median_update, 1e-12))
    return {"seed": arm["seed"], "optimizer": arm["optimizer"],
            "median_grad_norm": median_grad, "median_update_norm": median_update,
            "mean_step_s": statistics.mean(elapsed),
            "start_train_loss": events[0]["train_loss"],
            "end_train_loss": events[-1]["train_loss"], "numerical_anomaly": anomaly}


def _model_suggestion(suggestion, lrs, report_sha256):
    if suggestion is None:
        return {"status": "not_submitted"}
    if not isinstance(suggestion, dict) or set(suggestion) != {
        "optimizer", "learning_rate", "base_sha256", "report_sha256", "reason"
    }:
        return {"status": "rejected", "reason": "invalid_schema"}
    name, lr, reason = suggestion["optimizer"], suggestion["learning_rate"], suggestion["reason"]
    if (type(name) is not str or name not in lrs or suggestion["base_sha256"] != BASE
            or suggestion["report_sha256"] != report_sha256):
        return {"status": "rejected", "reason": "target_mismatch"}
    if (not _finite(lr) or type(reason) is not str or not 1 <= len(reason) <= 240
            or any(ord(char) < 32 or ord(char) == 127 for char in reason)):
        return {"status": "rejected", "reason": "invalid_value"}
    low, high = LIMITS[name]
    if not low <= lr <= high or not 0.8 * lrs[name] <= lr <= 1.2 * lrs[name]:
        return {"status": "rejected", "reason": "unbounded_lr"}
    return {"status": "bounded_research_suggestion_only", "optimizer": name,
            "learning_rate": lr, "muon_lora_unvalidated": name == "muon",
            "trial_authorized": False}


def observe(report, suggestion=None):
    verified = review(report)
    if verified.get("status") != "review_required":
        return blocked("source_review_failed:" + verified.get("reason", "unknown"))
    observations = [_summarize(arm) for arm in report["arms"]]
    if any(x["numerical_anomaly"] for x in observations):
        return blocked("numerical_telemetry_anomaly")
    if len(report["seeds"]) < 3 or report["steps"] < 20:
        return blocked("insufficient_replicates_or_steps")
    # A content digest binds an advisory proposal to exactly this telemetry; it
    # is not a signature or proof of authenticity. Signed receipts are required
    # before any future trial admission.
    report_sha256 = hashlib.sha256(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    lrs = {}
    for name in ("adamw", "muon"):
        values = {arm["learning_rate"] for arm in report["arms"]
                  if arm["optimizer"] == name}
        if len(values) != 1:
            return blocked("unpaired_learning_rates")
        lrs[name] = next(iter(values))
    grid = []
    for name, baseline in lrs.items():
        lo, hi = LIMITS[name]
        for factor in (0.8, 1.0, 1.2):
            value = baseline * factor
            if lo <= value <= hi:
                grid.append({"optimizer": name, "learning_rate": value,
                             "research_only": True, "muon_lora_unvalidated": name == "muon"})
    return {"schema": SCHEMA, "status": "research_observation_only",
            "model": report["model"], "base_sha256": BASE,
            "report_sha256": report_sha256, "report_sha_is_signature": False,
            "sample_type": "synthetic_random_tokens", "observations": observations,
            "proposed_next_experiment_grid": grid,
            "model_suggestion_review": _model_suggestion(suggestion, lrs, report_sha256),
            "fresh_independent_holdout_required": True,
            "real_task_quality_improvement_established": False,
            "optimizer_switch_permitted": False,
            "trial_authorized": False, "weight_promotion_permitted": False}


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_json_key")
        output[key] = value
    return output


def _read_json(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("unsafe_input")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_report", type=Path)
    parser.add_argument("--model-suggestion", type=Path)
    args = parser.parse_args()
    try:
        report = _read_json(args.telemetry_report)
        proposal = _read_json(args.model_suggestion) if args.model_suggestion else None
        result = observe(report, proposal)
    except (OSError, ValueError, UnicodeError, TypeError, KeyError) as exc:
        result = blocked("invalid_input:" + type(exc).__name__)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

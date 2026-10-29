"""Read-only, bounded training telemetry observer for LFM3 optimizer experiments.

Accepts numeric metadata only. Never controls a trainer or promotes weights.
"""

import argparse
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "sensorgraft.lfm3.optimizer-step.v1"
MAX_BYTES = 5_000_000
MAX_EVENTS = 10_000
MIN_STEPS = 20
FIELDS = frozenset(
    {
        "schema",
        "run_id",
        "step",
        "seed",
        "base_sha256",
        "dataset_sha256",
        "optimizer",
        "learning_rate",
        "train_loss",
        "grad_norm",
        "update_norm",
        "elapsed_s",
    }
)
BOUNDS = {"adamw": (1e-6, 2e-4), "muon": (1e-5, 2e-3)}
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
HEX32 = re.compile(r"[a-f0-9]{32}\Z")


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _blocked(reason):
    return {
        "schema": "sensorgraft.lfm3.optimizer-observation.v1",
        "status": "blocked",
        "reason": reason,
        "proposal": None,
        "weight_promotion_permitted": False,
    }


def _event_error(e):
    if not isinstance(e, dict) or set(e) != FIELDS:
        return "unexpected_or_missing_fields"
    if (
        e["schema"] != SCHEMA
        or not isinstance(e["run_id"], str)
        or not HEX32.fullmatch(e["run_id"])
    ):
        return "invalid_schema_or_run_id"
    if type(e["step"]) is not int or e["step"] < 0 or type(e["seed"]) is not int or e["seed"] < 0:
        return "invalid_step_or_seed"
    if any(
        not isinstance(e[k], str) or not HEX64.fullmatch(e[k])
        for k in ("base_sha256", "dataset_sha256")
    ):
        return "invalid_provenance_digest"
    name = e["optimizer"]
    if not isinstance(name, str) or name not in BOUNDS:
        return "unknown_optimizer"
    for k in ("learning_rate", "train_loss", "grad_norm", "update_norm", "elapsed_s"):
        if not _finite(e[k]):
            return "nonfinite_metric"
    if not BOUNDS[name][0] <= e["learning_rate"] <= BOUNDS[name][1]:
        return "unsafe_learning_rate"
    if min(e["train_loss"], e["grad_norm"], e["update_norm"]) < 0 or e["elapsed_s"] <= 0:
        return "invalid_metric_range"
    return None


def observe(events):
    """Return an offline suggestion, never a command to apply it."""
    if not isinstance(events, list) or len(events) > MAX_EVENTS or not events:
        return _blocked("invalid_event_count")
    fixed = ("run_id", "seed", "base_sha256", "dataset_sha256", "optimizer")
    prior = None
    for idx, e in enumerate(events):
        error = _event_error(e)
        if error:
            return _blocked(error)
        if prior is not None:
            if any(e[k] != prior[k] for k in fixed):
                return _blocked("mixed_experiment_provenance")
            if e["step"] != prior["step"] + 1:
                return _blocked("duplicate_or_missing_step")
        prior = e
    if len(events) < MIN_STEPS:
        return _blocked("insufficient_contiguous_steps")
    early = events[:5]
    late = events[-5:]
    early_loss = statistics.median(x["train_loss"] for x in early)
    late_loss = statistics.median(x["train_loss"] for x in late)
    early_grad = statistics.median(x["grad_norm"] for x in early)
    late_grad = statistics.median(x["grad_norm"] for x in late)
    if early_loss <= 0 or early_grad <= 0:
        return _blocked("zero_reference_metric")
    grad_ratio = late_grad / early_grad
    if grad_ratio >= 10 or late_loss > early_loss * 2:
        return _blocked("severe_training_instability")
    last = events[-1]
    # A proposal is an experiment suggestion, not evidence of generalization.
    reason = None
    factor = 1.0
    if grad_ratio >= 3 or late_loss > early_loss * 1.10:
        reason, factor = "gradient_or_loss_instability", 0.5
    elif late_loss >= early_loss * 0.99:
        reason, factor = "training_loss_plateau", 0.9
    status = "review_required" if reason else "observed"
    proposal = None
    if reason:
        low, high = BOUNDS[last["optimizer"]]
        candidate = max(low, min(high, last["learning_rate"] * factor))
        if candidate < last["learning_rate"]:
            proposal = {
                "optimizer": last["optimizer"],
                "learning_rate": candidate,
                "reason": reason,
                "requires_fresh_holdout_evaluation": True,
            }
    return {
        "schema": "sensorgraft.lfm3.optimizer-observation.v1",
        "status": status if proposal else "observed",
        "reason": reason if proposal else "no_bounded_adjustment",
        "proposal": proposal,
        "weight_promotion_permitted": False,
        "summary": {
            "run_id": last["run_id"],
            "step_count": len(events),
            "step_first": events[0]["step"],
            "step_last": last["step"],
            "train_loss_early_median": early_loss,
            "train_loss_late_median": late_loss,
            "grad_norm_ratio": grad_ratio,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("events_jsonl", type=Path)
    args = p.parse_args()
    try:
        if args.events_jsonl.stat().st_size > MAX_BYTES:
            result = _blocked("input_size_limit")
        else:
            with args.events_jsonl.open(encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            result = observe(events)
    except (OSError, ValueError, UnicodeError):
        result = _blocked("invalid_jsonl")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
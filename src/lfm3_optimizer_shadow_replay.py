"""Deterministic shadow replay of feedback-controlled LR, with no weight writes.

Decisions use already-verified training-step telemetry only; there is no
lookahead, optimizer object, live model, checkpoint or training execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from src.lfm3_optimizer_observer import LIMITS, _read_json, blocked, observe

SCHEMA = "sensorgraft.lfm3.lr-shadow-replay.v1"
WINDOW = 5
COOLDOWN = 5
MAX_CHANGES_PER_ARM = 3
MAX_RATIO_FROM_START = 0.20


def _decision(window):
    """Simple experimental controller, not an optimizer quality claim."""
    losses = [e["train_loss"] for e in window]
    gradients = [e["grad_norm"] for e in window]
    updates = [e["update_norm"] for e in window]
    if (gradients[-1] > 3 * max(statistics.median(gradients[:-1]), 1e-12)
            or updates[-1] > 3 * max(statistics.median(updates[:-1]), 1e-12)
            or losses[-1] > losses[0] * 1.02):
        return "reduce_lr", 0.90
    if losses[-1] >= losses[0] * 0.9975:
        return "plateau_probe", 1.05
    return "hold", 1.0


def _simulate_arm(arm, initial_lr):
    name = arm["optimizer"]
    baseline = arm["learning_rate"]
    lo, hi = LIMITS[name]
    floor = max(lo, initial_lr * (1 - MAX_RATIO_FROM_START),
                baseline * (1 - MAX_RATIO_FROM_START))
    ceiling = min(hi, initial_lr * (1 + MAX_RATIO_FROM_START),
                  baseline * (1 + MAX_RATIO_FROM_START))
    if not floor <= initial_lr <= ceiling:
        raise ValueError("initial_learning_rate_outside_shadow_range")
    current = initial_lr
    decisions = []
    for end in range(WINDOW, len(arm["events"]), COOLDOWN):
        if len(decisions) >= MAX_CHANGES_PER_ARM:
            break
        window = arm["events"][end - WINDOW:end]
        action, factor = _decision(window)
        next_lr = min(ceiling, max(floor, current * factor))
        # Exact no-op holds, no feigned learning-rate updates.
        if math.isclose(next_lr, current, rel_tol=1e-14):
            action, next_lr = "hold", current
        decisions.append({
            "effective_from_step": end, "observed_steps": [end - WINDOW, end - 1],
            "action": action, "previous_lr": current, "proposed_lr": next_lr,
            "applied_to_training": False,
        })
        current = next_lr
    return {"seed": arm["seed"], "optimizer": name, "initial_lr": initial_lr,
            "shadow_final_lr": current, "decisions": decisions,
            "weight_updates_performed": 0}


def replay(report, suggestion=None):
    observation = observe(report, suggestion)
    if observation["status"] != "research_observation_only":
        return blocked("source_observer_denied")
    accepted = observation["model_suggestion_review"]
    if suggestion is not None and accepted.get("status") != "bounded_research_suggestion_only":
        return blocked("untrusted_optimizer_suggestion")
    selected = accepted.get("optimizer") if suggestion is not None else None
    rows = []
    try:
        for arm in report["arms"]:
            if selected is not None and arm["optimizer"] != selected:
                continue
            lr = accepted["learning_rate"] if selected is not None else arm["learning_rate"]
            rows.append(_simulate_arm(arm, lr))
    except (ValueError, TypeError, KeyError, OverflowError):
        return blocked("invalid_shadow_configuration")
    if not rows:
        return blocked("empty_shadow_replay")
    result = {
        "schema": SCHEMA, "status": "shadow_replay_only",
        "base_sha256": observation["base_sha256"],
        "source_report_sha256": observation["report_sha256"],
        "source_authenticated": False,
        "synthetic_token_training_only": True,
        "uses_heldout_for_lr_decisions": False,
        "per_seed_arms": rows,
        "selected_by_model": selected is not None,
        "model_response_executed": False,
        "real_training_executed": False,
        "improvement_established": False,
        "training_configuration_applied": False,
        "trial_authorized": False, "weight_promotion_permitted": False,
    }
    result["replay_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--model-suggestion", type=Path)
    args = parser.parse_args()
    try:
        report = _read_json(args.report)
        suggestion = _read_json(args.model_suggestion) if args.model_suggestion else None
        result = replay(report, suggestion)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        result = blocked("invalid_input:" + type(exc).__name__)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

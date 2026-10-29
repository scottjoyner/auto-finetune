"""Strict, descriptive-only three-point AdamW/Muon LFM3 LoRA grid review.

No model selection, optimizer mutation, training or checkpoint promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from src.lfm3_optimizer_observer import _read_json, blocked
from src.lfm3_real_lora_review import review

SCHEMA = "sensorgraft.lfm3.adamw-muon-lrgrid-review.v1"
LR_LEVELS = (0.00008, 0.0001, 0.00012)


def analyze(reports):
    if not isinstance(reports, (list, tuple)) or len(reports) != 3:
        return blocked("expected_three_paired_optimizer_reports")
    for report in reports:
        if review(report).get("status") != "review_required":
            return blocked("source_report_failed_verification")
    anchor = reports[0]
    arms_by_key = defaultdict(list)
    run_ids = set()
    for report in reports:
        if (report["model"] != anchor["model"]
                or report["base_revision"] != anchor["base_revision"]
                or report["base_sha256"] != anchor["base_sha256"]
                or report["seeds"] != anchor["seeds"]
                or report["steps"] != anchor["steps"]
                or report["synthetic_only"] is not True
                or report["weight_promotion_permitted"] is not False):
            return blocked("cross_trial_provenance_mismatch")
        for arm in report["arms"]:
            key = (arm["seed"], arm["optimizer"])
            if arm["run_id"] in run_ids:
                return blocked("replayed_optimizer_run_id")
            run_ids.add(arm["run_id"])
            arms_by_key[key].append(arm)
    for key, arms in arms_by_key.items():
        if len(arms) != 3:
            return blocked("incomplete_seed_or_optimizer_grid")
        if len({a["learning_rate"] for a in arms}) != 3:
            return blocked("repeated_or_missing_lr")
        if {round(a["learning_rate"], 10) for a in arms} != set(LR_LEVELS):
            return blocked("unexpected_lr_grid")
        for field in ("initial_adapter_sha256", "train_sha256",
                      "heldout_sha256", "heldout_loss_before", "steps"):
            if len({a[field] for a in arms}) != 1:
                return blocked("unpaired_starting_state_between_lr_trials")
    if set(arms_by_key) != {(seed, opt) for seed in anchor["seeds"]
                           for opt in ("adamw", "muon")}:
        return blocked("unexpected_optimizer_or_seed_inventory")
    cells = []
    for optimizer in ("adamw", "muon"):
        for lr in LR_LEVELS:
            arms = [next(a for a in arms_by_key[(seed, optimizer)]
                         if round(a["learning_rate"], 10) == lr)
                    for seed in sorted(anchor["seeds"])]
            cells.append({
                "optimizer": optimizer,
                "learning_rate": lr,
                "seed_count": len(arms),
                "per_seed_final_synthetic_loss": [
                    {"seed": a["seed"], "heldout_loss": a["heldout_loss_after"]}
                    for a in arms
                ],
                "mean_final_synthetic_loss": statistics.mean(
                    a["heldout_loss_after"] for a in arms),
                "mean_train_step_s": statistics.mean(
                    e["elapsed_s"] for a in arms for e in a["events"]),
                "mean_grad_norm": statistics.mean(
                    e["grad_norm"] for a in arms for e in a["events"]),
                "mean_update_norm": statistics.mean(
                    e["update_norm"] for a in arms for e in a["events"]),
            })
    return {
        "schema": SCHEMA,
        "status": "descriptive_synthetic_grid_only",
        "model": anchor["model"],
        "base_sha256": anchor["base_sha256"],
        "runs_verified": len(run_ids),
        "optimizer_steps_verified": len(run_ids) * anchor["steps"],
        "lr_levels": list(LR_LEVELS),
        "cells": cells,
        "real_task_quality_measured": False,
        "independent_selection_holdout_available": False,
        "optimizer_or_lr_selected": None,
        "no_statistical_superiority_claim": True,
        "trial_authorized": False,
        "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=3)
    args = parser.parse_args()
    try:
        result = analyze([_read_json(p) for p in args.reports])
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        result = blocked("invalid_source:" + type(exc).__name__)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

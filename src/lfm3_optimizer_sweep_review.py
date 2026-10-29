"""Fail-closed, read-only cross-run reviewer for synthetic optimizer A/B trials.

The output is experiment prioritization only. It never changes an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from src.lfm3_optimizer_telemetry import observe


def blocked(reason):
    return {
        "schema": "sensorgraft.lfm3.synthetic-sweep-review.v1",
        "status": "blocked",
        "reason": reason,
        "proposal": None,
        "weight_promotion_permitted": False,
    }


def review(reports):
    if not isinstance(reports, list) or not 3 <= len(reports) <= 10:
        return blocked("invalid_report_count")
    expected_seeds = None
    expected_steps = None
    baseline = {}
    experiments = {}
    custody = []
    for report in reports:
        if not isinstance(report, dict) or any(
            report.get(k) != v
            for k, v in (
                ("schema", "sensorgraft.lfm3.synthetic-optimizer-ab.v1"),
                ("status", "synthetic_surrogate_only"),
                ("model", "tiny_cpu_mlp_not_lfm3"),
                ("real_private_traces_used", False),
                ("production_ready", False),
                ("weight_promotion_permitted", False),
            )
        ):
            return blocked("report_not_synthetic_or_promotion_safe")
        seeds = report.get("seeds")
        steps = report.get("steps_per_arm")
        if (
            not isinstance(seeds, list)
            or len(seeds) < 3
            or len(seeds) > 8
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or type(steps) is not int
            or not 20 <= steps <= 200
        ):
            return blocked("invalid_seed_or_step_custody")
        if expected_seeds is None:
            expected_seeds, expected_steps = set(seeds), steps
        if set(seeds) != expected_seeds or steps != expected_steps:
            return blocked("unpaired_experiment_schedule")
        arms = report.get("arms")
        if not isinstance(arms, list) or len(arms) != len(seeds) * 2:
            return blocked("invalid_arm_inventory")
        indexed = {}
        for arm in arms:
            if not isinstance(arm, dict):
                return blocked("invalid_arm")
            seed, name, lr = arm.get("seed"), arm.get("optimizer"), arm.get("learning_rate")
            if (
                type(seed) is not int
                or seed not in expected_seeds
                or name not in ("adamw", "muon")
                or type(lr) not in (int, float)
                or not math.isfinite(lr)
                or not (1e-6 <= lr <= 2e-4 if name == "adamw" else 1e-5 <= lr <= 2e-3)
                or (seed, name) in indexed
                or arm.get("steps") != steps
                or arm.get("weights_promoted") is not False
                or type(arm.get("adamw_fallback_lr")) not in (int, float)
                or not math.isfinite(arm["adamw_fallback_lr"])
                or not 1e-6 <= arm["adamw_fallback_lr"] <= 2e-4
            ):
                return blocked("unsafe_or_duplicate_arm")
            for k in ("base_sha256", "dataset_sha256", "holdout_sha256", "checkpoint_after_sha256"):
                h = arm.get(k)
                if (
                    not isinstance(h, str)
                    or len(h) != 64
                    or any(c not in "0123456789abcdef" for c in h)
                ):
                    return blocked("invalid_digest")
            for k in ("holdout_loss_before", "holdout_loss_after"):
                x = arm.get(k)
                if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                    return blocked("invalid_holdout_metric")
            if arm.get("heldout_disjoint") is not True:
                return blocked("holdout_not_disjoint")
            events = arm.get("events")
            if not isinstance(events, list) or len(events) != steps:
                return blocked("invalid_telemetry_length")
            check = observe(events)
            if check["status"] == "blocked":
                return blocked("untrusted_telemetry:" + check["reason"])
            if (
                events[0]["seed"] != seed
                or events[0]["optimizer"] != name
                or events[0]["dataset_sha256"] != arm["dataset_sha256"]
                or events[0]["base_sha256"] != arm["base_sha256"]
                or events[0]["learning_rate"] != lr
            ):
                return blocked("telemetry_custody_mismatch")
            indexed[(seed, name)] = arm
        if set(indexed) != {(s, name) for s in expected_seeds for name in ("adamw", "muon")}:
            return blocked("missing_optimizer_arm")
        for seed in expected_seeds:
            a, m = indexed[(seed, "adamw")], indexed[(seed, "muon")]
            equal_keys = ("base_sha256", "dataset_sha256", "holdout_sha256", "holdout_loss_before")
            if any(a[key] != m[key] for key in equal_keys):
                return blocked("mismatched_paired_arm")
            if (
                a["adamw_fallback_lr"] != a["learning_rate"]
                or m["adamw_fallback_lr"] != a["learning_rate"]
            ):
                return blocked("adamw_fallback_learning_rate_drift")
            ref = (
                a["base_sha256"],
                a["dataset_sha256"],
                a["holdout_sha256"],
                a["holdout_loss_before"],
                a["learning_rate"],
                a["holdout_loss_after"],
                a["checkpoint_after_sha256"],
            )
            if seed in baseline and baseline[seed] != ref:
                return blocked("baseline_drift_between_reports")
            baseline[seed] = ref
        lr_values = {indexed[(s, "muon")]["learning_rate"] for s in expected_seeds}
        if len(lr_values) != 1:
            return blocked("mixed_muon_learning_rates")
        lr = next(iter(lr_values))
        if lr in experiments:
            return blocked("duplicate_muon_setting")
        experiments[lr] = [
            indexed[(s, "muon")]["holdout_loss_after"] for s in sorted(expected_seeds)
        ]
        custody.append(
            [lr, [indexed[(s, "muon")]["checkpoint_after_sha256"] for s in sorted(expected_seeds)]]
        )
    ordered = sorted(experiments)
    records = [
        {
            "muon_learning_rate": lr,
            "mean_muon_holdout_loss": statistics.mean(experiments[lr]),
            "mean_adamw_holdout_loss": statistics.mean(
                baseline[s][5] for s in sorted(expected_seeds)
            ),
            "paired_muon_minus_adamw": [
                value - baseline[seed][5]
                for seed, value in zip(sorted(expected_seeds), experiments[lr])
            ],
        }
        for lr in ordered
    ]
    best = min(records, key=lambda r: (r["mean_muon_holdout_loss"], r["muon_learning_rate"]))
    next_lr = min(0.002, best["muon_learning_rate"] * 1.25)
    if next_lr <= best["muon_learning_rate"]:
        next_lr = None
    digest = hashlib.sha256(json.dumps(sorted(custody), sort_keys=True).encode()).hexdigest()
    return {
        "schema": "sensorgraft.lfm3.synthetic-sweep-review.v1",
        "status": "review_required",
        "model": "tiny_cpu_mlp_not_lfm3",
        "synthetic_only": True,
        "seed_count": len(expected_seeds),
        "steps_per_arm": expected_steps,
        "input_custody_sha256": digest,
        "results": records,
        "proposal": {
            "best_observed_muon_lr": best["muon_learning_rate"],
            "next_muon_lr_probe": next_lr,
            "requires_new_independent_heldout_evaluation": True,
            "may_change_running_optimizer": False,
        },
        "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    if any(p.stat().st_size > 2_000_000 for p in args.reports):
        result = blocked("oversized_report")
    else:
        try:
            result = review([json.loads(p.read_text(encoding="utf-8")) for p in args.reports])
        except (OSError, ValueError, UnicodeError):
            result = blocked("unreadable_report")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
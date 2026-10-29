"""Conservative three-seed advisory admission for LFM3 Dust predictors.

A learned prediction head must outperform the trivial zero-gain forecast and
its top-1 acquisition value on EACH original seed before it can be nominated
for a NEW independent study. Neither pass nor failure grants authority to
modify live weights, choose the deployed optimizer, or open the test split.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from src.lfm3_dust_prequential_review import review as review_source

SCHEMA = "sensorgraft.lfm3.dust-predictor-admission.v1"
POLICIES = ("direction", "activation", "calibrated")


def blocked(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "eligible_for_independent_confirmation": False,
            "optimizer_switch_permitted": False,
            "model_update_permitted": False,
            "weight_promotion_permitted": False}


def assess(reports, *, minimum_independent_episodes=32):
    if type(minimum_independent_episodes) is not int or minimum_independent_episodes != 32:
        return blocked("unregistered_independent_episode_requirement")
    if not isinstance(reports, list) or len(reports) != 3:
        return blocked("three_independent_seeds_required")
    reviewed, seeds, runs = [], set(), set()
    for source in reports:
        r = review_source(source)
        if r["status"] != "descriptive_review_required":
            return blocked("untrusted_candidate_evidence")
        seed, run = source["seed"], source["model_run_id"]
        if seed in seeds or run in runs:
            return blocked("duplicate_seed_or_run_id")
        seeds.add(seed)
        runs.add(run)
        reviewed.append(source)
    if seeds != {1, 2, 3}:
        return blocked("incorrect_seed_inventory")
    result = {}
    reasons = []
    for policy in POLICIES:
        pairs = []
        for report in sorted(reviewed, key=lambda r: r["seed"]):
            metrics = report["metrics"]
            ref, candidate = metrics["zero"], metrics[policy]
            mse_ratio = candidate["mse"] / ref["mse"] if ref["mse"] > 0 else math.inf
            gain_advantage = (candidate["mean_gain_if_top1_probed"]
                              - ref["mean_gain_if_top1_probed"])
            pairs.append({"seed": report["seed"],
                          "candidate_mse": candidate["mse"],
                          "zero_baseline_mse": ref["mse"],
                          "mse_ratio_vs_zero": mse_ratio,
                          "top1_gain_advantage_vs_zero": gain_advantage})
        mse_pass = all(
            p["mse_ratio_vs_zero"] <= .95 for p in pairs
        )
        gain_pass = all(
            p["top1_gain_advantage_vs_zero"] > 0 for p in pairs
        )
        result[policy] = {
            "per_seed": pairs, "mse_gate": mse_pass, "gain_gate": gain_pass,
            "mean_mse_ratio": statistics.mean(p["mse_ratio_vs_zero"] for p in pairs),
            "mean_top1_gain_advantage": statistics.mean(
                p["top1_gain_advantage_vs_zero"] for p in pairs),
            "eligible_for_independent_confirmation": False,
        }
        if not mse_pass:
            reasons.append(f"{policy}:forecast_mse_not_5pct_better_than_zero_on_every_seed")
        if not gain_pass:
            reasons.append(f"{policy}:top1_gain_not_better_than_zero_on_every_seed")
    # 3 seeds x 8 independent procedural training prompts; directions within
    # an episode are correlated observations, NOT independent source groups.
    groups = len(reviewed) * 8
    if groups < minimum_independent_episodes:
        reasons.append("insufficient_independent_source_episodes_24_lt_32")
    # Even passing the negative-control screen requires separately reviewed
    # new source groups with real tasks and prospective confirmation.
    return {
        "schema": SCHEMA,
        "status": "hold_not_qualified" if reasons else "eligible_for_separate_study_only",
        "blocking_reasons": reasons,
        "seed_count": 3, "unique_training_prompt_episodes": groups,
        "candidate_results": result,
        "prequential_evidence": True,
        "actual_training_or_weight_update_count": 0,
        "independent_test_evaluated": False,
        "eligible_for_independent_confirmation": not reasons,
        "optimizer_switch_permitted": False,
        "model_update_permitted": False,
        "weight_promotion_permitted": False,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs=3, type=Path)
    args = parser.parse_args()
    try:
        sources = []
        for path in args.evidence:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 150000:
                raise ValueError("unsafe_evidence_path")
            sources.append(json.loads(path.read_text(encoding="utf-8")))
        result = assess(sources)
    except (OSError, ValueError, UnicodeError, KeyError, TypeError, OverflowError):
        result = blocked("malformed_evidence")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Fail-closed offline verifier for procedural LFM3 text-task research reports.

Validates protocol and paired arm structure, not cryptographic run authenticity.
It never infers an optimizer winner or promotes checkpoint changes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from src.lfm3_lora_text_task_probe import FAMILIES, SCHEMA as SOURCE, _digest, dataset
from src.lfm3_real_lora_review import BASE

SCHEMA = "sensorgraft.lfm3.lora-text-task-review.v1"
HEX = set("0123456789abcdef")


def _hash(s, size=64):
    return type(s) is str and len(s) == size and all(c in HEX for c in s)


def blocked(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "weight_promotion_permitted": False, "trial_authorized": False}


def _finite(n, low=0, high=1e6):
    return type(n) in (int, float) and math.isfinite(n) and low <= n <= high


def _scores_valid(scores):
    if not isinstance(scores, dict) or set(scores) != {
        "correct", "total", "mean_correct_minus_foil_logprob", "families"
    }:
        return False
    if (type(scores["total"]) is not int or scores["total"] != 9
            or type(scores["correct"]) is not int
            or not 0 <= scores["correct"] <= 9
            or not _finite(scores["mean_correct_minus_foil_logprob"], -1e5, 1e5)):
        return False
    fam = scores["families"]
    return (isinstance(fam, dict) and set(fam) == set(FAMILIES)
            and all(isinstance(v, dict) and set(v) == {"correct", "total"}
                    and type(v["correct"]) is int and type(v["total"]) is int
                    and v["total"] == 3 and 0 <= v["correct"] <= 3
                    for v in fam.values())
            and sum(v["correct"] for v in fam.values()) == scores["correct"])


def review(report):
    expected = {
        "schema": SOURCE, "status": "research_text_task_diagnostic_only",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE,
        "data_kind": "procedural_public_arithmetic_and_parity_text",
        "answer_metric": "two_choice_teacher_forced_logprob",
        "selection_used_for_reporting_not_optimizer_choice": True,
        "untouched_test_evaluated": False,
        "real_task_generalization_established": False,
        "weight_promotion_permitted": False, "trial_authorized": False,
    }
    if not isinstance(report, dict) or any(report.get(k) != v for k, v in expected.items()):
        return blocked("unverified_source_or_promotion_claim")
    seeds, steps, arms = report.get("seeds"), report.get("steps"), report.get("arms")
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3
            or any(type(s) is not int or not 0 <= s <= 2**32 - 1 for s in seeds)
            or len(set(seeds)) != len(seeds)
            or type(steps) is not int or not 1 <= steps <= 20
            or not isinstance(arms, list) or len(arms) != 2 * len(seeds)):
        return blocked("incorrect_seeds_steps_or_arms")
    manifests = report.get("dataset_sha256_by_seed")
    if not isinstance(manifests, dict) or set(manifests) != {str(s) for s in seeds}:
        return blocked("invalid_dataset_manifests")
    for seed in seeds:
        recorded = manifests[str(seed)]
        actual = {k: _digest(v) for k, v in dataset(seed).items()}
        if not isinstance(recorded, dict) or recorded != actual:
            return blocked("dataset_partition_identity_mismatch")
    by_seed, run_ids = {}, set()
    for arm in arms:
        if not isinstance(arm, dict) or set(arm) != {
            "run_id", "seed", "optimizer", "learning_rate", "trainable_parameters",
            "initial_adapter_sha256", "final_adapter_sha256", "selection_before",
            "selection_after", "events", "weight_promotion_permitted"
        }:
            return blocked("unexpected_arm_schema")
        seed, name, lr = arm["seed"], arm["optimizer"], arm["learning_rate"]
        if (type(seed) is not int or seed not in seeds
                or type(name) is not str or name not in ("adamw", "muon")
                or not _finite(lr, 1e-6, 2e-3)
                or not (lr <= 2e-4 if name == "adamw" else lr >= 1e-5)
                or arm["trainable_parameters"] != 146432
                or not _hash(arm["run_id"], size=32)
                or arm["run_id"] in run_ids
                or not _hash(arm["initial_adapter_sha256"])
                or not _hash(arm["final_adapter_sha256"])
                or arm["initial_adapter_sha256"] == arm["final_adapter_sha256"]
                or arm["weight_promotion_permitted"] is not False
                or not _scores_valid(arm["selection_before"])
                or not _scores_valid(arm["selection_after"])):
            return blocked("invalid_optimizer_arm_or_scores")
        run_ids.add(arm["run_id"])
        events = arm["events"]
        if not isinstance(events, list) or len(events) != steps:
            return blocked("incomplete_training_steps")
        for step, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {
                "step", "optimizer", "learning_rate", "train_loss",
                "unclipped_grad_norm", "grad_clip_limit", "update_norm", "elapsed_s"
            }:
                return blocked("invalid_event_schema")
            if (type(event["step"]) is not int or event["step"] != step
                    or event["optimizer"] != name or event["learning_rate"] != lr
                    or event["grad_clip_limit"] != 1
                    or not _finite(event["train_loss"], 1e-12)
                    or not _finite(event["unclipped_grad_norm"])
                    or not _finite(event["update_norm"])
                    or not _finite(event["elapsed_s"], 1e-12, 3600)):
                return blocked("invalid_or_unbounded_training_telemetry")
        by_seed.setdefault(seed, {})
        if name in by_seed[seed]:
            return blocked("duplicate_optimizer_for_seed")
        by_seed[seed][name] = arm
    if any(set(by_seed.get(s, {})) != {"adamw", "muon"} for s in seeds):
        return blocked("missing_paired_optimizer")
    pairs = []
    for seed in sorted(seeds):
        a, m = by_seed[seed]["adamw"], by_seed[seed]["muon"]
        if (a["initial_adapter_sha256"] != m["initial_adapter_sha256"]
                or a["selection_before"] != m["selection_before"]):
            return blocked("unpaired_adapter_or_baseline_scores")
        pairs.append({
            "seed": seed,
            "initial_selection_correct": a["selection_before"]["correct"],
            "adamw_selection_correct": a["selection_after"]["correct"],
            "muon_selection_correct": m["selection_after"]["correct"],
            "adamw_accuracy_delta": (a["selection_after"]["correct"]
                                     - a["selection_before"]["correct"]) / 9,
            "muon_accuracy_delta": (m["selection_after"]["correct"]
                                    - m["selection_before"]["correct"]) / 9,
        })
    return {
        "schema": SCHEMA, "status": "review_required",
        "source_authenticated": False,
        "model": report["model"], "seeds": seeds, "steps_per_arm": steps,
        "true_optimizer_steps": len(arms) * steps,
        "task_type": "procedural_arithmetic_text_pairwise_choice_only",
        "pairs": pairs,
        "mean_adamw_correct": statistics.mean(p["adamw_selection_correct"] for p in pairs),
        "mean_muon_correct": statistics.mean(p["muon_selection_correct"] for p in pairs),
        "untouched_test_evaluated": False,
        "independent_real_task_quality_proven": False,
        "optimizer_selection_permitted": False,
        "weight_promotion_permitted": False, "trial_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        if (args.input.is_symlink() or not args.input.is_file()
                or args.input.stat().st_size > 200_000):
            result = blocked("unsafe_or_oversized_file")
        else:
            result = review(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError, OverflowError) as e:
        result = blocked("invalid_input:" + type(e).__name__)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

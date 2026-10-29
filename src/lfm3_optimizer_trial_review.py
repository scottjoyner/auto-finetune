"""Verify externally evaluated optimizer trials without granting deployment.

Requires a key-held independent evaluator's HMAC receipt. A SHA-256 link to
telemetry is *not* authentication, and HMAC verifies key possession, not
reviewer independence or task validity. This module never runs training.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
from pathlib import Path

from src.lfm3_optimizer_observer import _read_json, blocked, observe

SCHEMA = "sensorgraft.lfm3.paired-optimizer-evaluation.v1"
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
HEX32 = re.compile(r"[a-f0-9]{32}\Z")
METRICS = {"task_correct", "task_total", "retention_correct", "retention_total",
           "safety_regressions", "tokens_per_s"}


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _valid_digest(value):
    return type(value) is str and HEX64.fullmatch(value) is not None


def _metrics_valid(m):
    return (
        isinstance(m, dict) and set(m) == METRICS
        and all(type(m[k]) is int for k in (
            "task_correct", "task_total", "retention_correct",
            "retention_total", "safety_regressions"))
        and 40 <= m["task_total"] <= 100_000
        and 40 <= m["retention_total"] <= 100_000
        and 0 <= m["task_correct"] <= m["task_total"]
        and 0 <= m["retention_correct"] <= m["retention_total"]
        and 0 <= m["safety_regressions"] <= m["task_total"]
        and type(m["tokens_per_s"]) in (int, float)
        and 0 < m["tokens_per_s"] <= 1e6
    )


def review_trial(report, proposal, receipt, hmac_key):
    """Read only; manual review is required even with a good authenticated receipt."""
    shadow = observe(report, proposal)
    if shadow.get("status") != "research_observation_only":
        return blocked("invalid_training_observation")
    advice = shadow["model_suggestion_review"]
    if advice.get("status") != "bounded_research_suggestion_only":
        return blocked("unverified_model_proposal")
    if type(hmac_key) is not bytes or len(hmac_key) < 32:
        return blocked("missing_independent_evaluator_key")
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema", "base_sha256", "report_sha256", "train_split_sha256",
        "selection_split_sha256", "eval_run_id", "source", "evaluations",
        "hmac_sha256"
    }:
        return blocked("invalid_receipt_schema")
    signature = receipt["hmac_sha256"]
    if not _valid_digest(signature):
        return blocked("invalid_receipt_signature")
    unsigned = {k: v for k, v in receipt.items() if k != "hmac_sha256"}
    try:
        raw = _canonical(unsigned)
    except (TypeError, ValueError):
        return blocked("unserializable_receipt")
    actual = hmac.new(hmac_key, raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, actual):
        return blocked("receipt_authentication_failed")
    if (receipt["schema"] != SCHEMA
            or receipt["source"] != "independent-isolated-evaluator"
            or receipt["base_sha256"] != shadow["base_sha256"]
            or receipt["report_sha256"] != shadow["report_sha256"]
            or not type(receipt["eval_run_id"]) is str
            or not HEX32.fullmatch(receipt["eval_run_id"])):
        return blocked("evaluation_custody_mismatch")
    train_hash = receipt["train_split_sha256"]
    selection_hash = receipt["selection_split_sha256"]
    original_hashes = {arm[k] for arm in report["arms"]
                       for k in ("train_sha256", "heldout_sha256")}
    if (not _valid_digest(train_hash) or not _valid_digest(selection_hash)
            or train_hash == selection_hash
            or train_hash in original_hashes or selection_hash in original_hashes):
        return blocked("evaluation_split_not_independent")
    evaluations = receipt["evaluations"]
    if not isinstance(evaluations, list) or len(evaluations) != len(report["seeds"]):
        return blocked("invalid_evaluation_inventory")
    seen, comparisons = set(), []
    for row in evaluations:
        if not isinstance(row, dict) or set(row) != {
            "seed", "baseline_run_id", "candidate_run_id",
            "candidate_optimizer", "candidate_learning_rate",
            "baseline", "candidate"
        }:
            return blocked("invalid_paired_evaluation")
        seed = row["seed"]
        if (type(seed) is not int or seed not in report["seeds"] or seed in seen
                or row["candidate_optimizer"] != advice["optimizer"]
                or type(row["candidate_learning_rate"]) not in (float, int)
                or row["candidate_learning_rate"] != advice["learning_rate"]):
            return blocked("unpaired_optimizer_or_seed")
        seen.add(seed)
        ids = (row["baseline_run_id"], row["candidate_run_id"])
        if any(type(x) is not str or not HEX32.fullmatch(x) for x in ids):
            return blocked("unverified_evaluation_run_id")
        if ids[0] == ids[1]:
            return blocked("same_evaluation_run_for_both_arms")
        a, b = row["baseline"], row["candidate"]
        if not _metrics_valid(a) or not _metrics_valid(b):
            return blocked("invalid_quality_or_safety_metrics")
        if a["task_total"] != b["task_total"] or a["retention_total"] != b["retention_total"]:
            return blocked("unpaired_task_evaluation")
        comparisons.append({
            "seed": seed,
            "task_accuracy_delta": b["task_correct"] / b["task_total"] - a["task_correct"] / a["task_total"],
            "retention_delta": b["retention_correct"] / b["retention_total"] - a["retention_correct"] / a["retention_total"],
            "new_safety_regressions": max(0, b["safety_regressions"] - a["safety_regressions"]),
            "throughput_ratio": b["tokens_per_s"] / a["tokens_per_s"],
        })
    if seen != set(report["seeds"]):
        return blocked("incomplete_seed_comparison")
    regression = any(
        c["task_accuracy_delta"] < -0.02
        or c["retention_delta"] < -0.02
        or c["new_safety_regressions"] > 0
        or c["throughput_ratio"] < 0.85
        for c in comparisons
    )
    return {
        "schema": "sensorgraft.lfm3.paired-optimizer-review.v1",
        "status": "manual_review_required",
        "receipt_hmac_verified": True,
        "independent_licensing_and_review_not_proven": True,
        "experiment": "nonproduction_candidate_vs_baseline",
        "comparisons": sorted(comparisons, key=lambda v: v["seed"]),
        "provisional_regression_flag": regression,
        "statistical_significance_established": False,
        "final_test_set_used": False,
        "trial_authorized": False, "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_report", type=Path)
    parser.add_argument("model_proposal", type=Path)
    parser.add_argument("evaluation_receipt", type=Path)
    args = parser.parse_args()
    key = os.environ.get("LFM3_EVAL_HMAC_KEY", "").encode()
    try:
        result = review_trial(_read_json(args.training_report),
                              _read_json(args.model_proposal),
                              _read_json(args.evaluation_receipt), key)
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as e:
        result = blocked("unreadable_input:" + type(e).__name__)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

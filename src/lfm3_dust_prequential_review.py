"""Fail-closed, CPU-free verifier for real LFM3 Dust all-candidate forecast audits.

Recomputes metrics from frozen pre-probe forecasts and observed per-candidate
scalar gains; requires all +/- directions to be scored before head updates.
This is STRUCTURAL verification only, not independent cryptographic execution
attestation or proof that model internals were really frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_prequential_audit import (
    CANDIDATES, EPISODES, POLICIES, SCHEMA as SOURCE, SIGMA, _metrics,
)
from src.lfm3_lora_text_task_probe import dataset, _digest
from src.lfm3_real_lora_review import BASE

SCHEMA = "sensorgraft.lfm3.dust-prequential-review.v1"
H64 = re.compile(r"[0-9a-f]{64}\Z")
H32 = re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "optimizer_policy_promotion_permitted": False,
            "weight_promotion_permitted": False}


def _finite(x, low=-100, high=100):
    return type(x) in (int, float) and math.isfinite(x) and low <= x <= high


def review(report):
    if not isinstance(report, dict):
        return blocked("invalid_report")
    seed = report.get("seed")
    if type(seed) is not int or seed not in (1,2,3,4,5,6,7,8,9):
        return blocked("unapproved_seed")
    expected_episodes = EPISODES if seed in (1,2,3) else 12
    expected = {
        "schema": SOURCE, "status": "real_lfm3_prequential_prediction_only",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE,
        "episodes": expected_episodes, "candidates_per_episode": CANDIDATES,
        "sigma": SIGMA, "parameter_scope": "last_two_attention_vproj_lora_B_only",
        "observed_parameter_count": 2048, "metrics_exclude_cold_start": True,
        "total_perturbed_forwards": 2 * CANDIDATES * expected_episodes,
        "total_context_forwards": expected_episodes,
        "total_backward_calls": 0, "model_adapter_update_count": 0,
        "independent_test_evaluated": False,
        "selection_partition_evaluated": False,
        "private_traces_used": False, "source_independently_authenticated": False,
        "equal_compute_to_adamw_muon": False,
        "model_quality_improvement_proven": False,
        "calibration_useful_for_generalization_proven": False,
        "weight_promotion_permitted": False, "trial_authorized": False,
    }
    if any(report.get(k) != v for k, v in expected.items()):
        return blocked("unverified_run_or_false_promotion_claim")
    if (type(report.get("model_run_id")) is not str
            or not H32.fullmatch(report["model_run_id"])
            or not _finite(report.get("elapsed_s"), 1e-10, 3600)):
        return blocked("missing_runtime_or_run_identity")
    initial, final = report.get("initial_adapter_sha256"), report.get("final_adapter_sha256")
    if (type(initial) is not str or not H64.fullmatch(initial)
            or final != initial):
        return blocked("adapter_checksum_changed_or_invalid")
    parts = dataset(seed)
    if report.get("data_manifest") != {name: _digest(rows) for name, rows in parts.items()}:
        return blocked("dataset_partition_provenance_mismatch")
    episodes = report.get("episode_receipts")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        return blocked("missing_episode_receipts")
    previous_weights = None
    for ep, row in enumerate(episodes):
        if (not isinstance(row, dict)
                or set(row) != {
                    "episode", "train_example_sha256", "context_sha256",
                    "train_clean_loss", "calibrated_weights_after_feedback",
                    "calibration_event_count_after", "candidates",
                    "perturbed_forward_count", "prompt_only_forward_count",
                    "training_backward_count"
                } or type(row["episode"]) is not int or row["episode"] != ep
                or row["perturbed_forward_count"] != 2 * CANDIDATES
                or row["prompt_only_forward_count"] != 1
                or row["training_backward_count"] != 0
                or not _finite(row["train_clean_loss"], 1e-8, 1000)):
            return blocked("malformed_episode")
        item = {k: v for k, v in parts["train"][ep].items() if k != "input_key"}
        item_sha = hashlib.sha256(json.dumps(
            item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (row["train_example_sha256"] != item_sha
                or type(row["context_sha256"]) is not str
                or not H64.fullmatch(row["context_sha256"])):
            return blocked("wrong_training_example_or_context_digest")
        cases = row["candidates"]
        if not isinstance(cases, list) or len(cases) != 2 * CANDIDATES:
            return blocked("incomplete_candidate_coverage")
        baseline_weights = None
        for pos, candidate in enumerate(cases):
            idx, sign = divmod(pos, 2)
            sign = 1 if sign == 0 else -1
            if (not isinstance(candidate, dict)
                    or set(candidate) != {
                        "candidate", "sign", "direction_coefficients",
                        "predictions", "direction_evidence",
                        "activation_evidence", "calibrated_evidence",
                        "head_weights_preprobe", "measured_gain"
                    } or type(candidate["candidate"]) is not int
                    or candidate["candidate"] != idx
                    or type(candidate["sign"]) is not int
                    or candidate["sign"] != sign
                    or not _finite(candidate["measured_gain"])):
                return blocked("mismatched_direction_or_gain")
            coeffs = candidate["direction_coefficients"]
            if (not isinstance(coeffs, list) or len(coeffs) != 8
                    or any(not _finite(x, -2, 2) for x in coeffs)
                    or abs(sum(float(x*x) for x in coeffs)-1) > 1e-5):
                return blocked("invalid_normalized_direction")
            if sign == -1 and any(abs(x + y) > 1e-6 for x, y in zip(
                    coeffs, cases[pos-1]["direction_coefficients"], strict=True)):
                return blocked("antithetic_direction_not_paired")
            if (candidate["direction_evidence"] != min(32, 2*CANDIDATES*ep)
                    or candidate["activation_evidence"] != min(32, 2*CANDIDATES*ep)
                    or candidate["calibrated_evidence"] != min(32, 2*CANDIDATES*ep)):
                return blocked("same_episode_result_used_as_prediction")
            forecasts = candidate["predictions"]
            if (not isinstance(forecasts, dict) or set(forecasts) != set(POLICIES)
                    or any(not _finite(v) for v in forecasts.values())
                    or (ep == 0 and any(v != 0 for v in forecasts.values()))
                    or forecasts["zero"] != 0):
                return blocked("unbounded_or_noncausal_forecast")
            weights = candidate["head_weights_preprobe"]
            if (not isinstance(weights, list) or len(weights) != 2
                    or not all(_finite(v, .05-1e-5, .95+1e-5) for v in weights)
                    or abs(sum(weights)-1) > 1e-7):
                return blocked("invalid_attention_gate_weights")
            if baseline_weights is None:
                baseline_weights = weights
            elif weights != baseline_weights:
                return blocked("probe_outcomes_changed_same_episode_ranking")
            if previous_weights is not None and weights != previous_weights:
                return blocked("preprobe_weights_not_from_prior_episode")
        end_weights = row["calibrated_weights_after_feedback"]
        if (not isinstance(end_weights, list) or len(end_weights) != 2
                or not all(_finite(v, .05-1e-5, .95+1e-5) for v in end_weights)
                or abs(sum(end_weights)-1) > 1e-7):
            return blocked("invalid_final_gate_weights")
        if (type(row["calibration_event_count_after"]) is not int
                or row["calibration_event_count_after"] != (0 if ep == 0 else 8*ep)):
            return blocked("unverified_calibration_episode_count")
        previous_weights = end_weights
    try:
        computed = _metrics(episodes, expected_episodes=expected_episodes)
    except (ValueError, TypeError, KeyError, OverflowError):
        return blocked("invalid_prequential_metric_reconstruction")
    recorded = report.get("metrics")
    if not isinstance(recorded, dict) or set(recorded) != set(POLICIES):
        return blocked("missing_metric_baseline")
    for policy, rec in recorded.items():
        if not isinstance(rec, dict) or set(rec) != set(computed[policy]):
            return blocked("metric_schema_mismatch")
        for key, expected_value in computed[policy].items():
            value = rec[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or abs(value-expected_value) > 1e-9):
                return blocked("metric_misreported_or_holdout_contamination")
    return {
        "schema": SCHEMA, "status": "descriptive_review_required",
        "seed": seed, "training_episodes": expected_episodes,
        "prequential_cases_per_policy": computed["zero"]["cases"],
        "metric_mse": {name: computed[name]["mse"] for name in POLICIES},
        "gain_per_top1_candidate": {
            name: computed[name]["mean_gain_if_top1_probed"] for name in POLICIES
        },
        "adapter_unchanged": True,
        "model_quality_improvement_proven": False,
        "independent_execution_attestation": False,
        "optimizer_policy_promotion_permitted": False,
        "weight_promotion_permitted": False,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    try:
        if (args.source.is_symlink() or not args.source.is_file()
                or args.source.stat().st_size > 150000):
            output = blocked("unsafe_input_file")
        else:
            output = review(json.loads(args.source.read_text(encoding="utf-8")))
    except (ValueError, OSError, UnicodeError, OverflowError):
        output = blocked("unreadable_input")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

"""Untrusted trial receipt review; all test receipts are synthetic and isolated."""
import copy
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_observer import BASE, observe
from src.lfm3_optimizer_trial_review import SCHEMA, _canonical, review_trial

KEY = b"isolated-evaluator-test-key-32-bytes-unique"
EVIDENCE = Path(__file__).resolve().parents[1] / (
    "research/evidence/lfm3-real-lora-20261029/"
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def model_proposal(report):
    return {"optimizer": "adamw", "learning_rate": 9e-5, "base_sha256": BASE,
            "report_sha256": observe(report)["report_sha256"],
            "reason": "Compare against fixed AdamW control, independent selection split"}


def metrics(correct=37, retention=45, safety=0, throughput=20):
    return {"task_correct": correct, "task_total": 50,
            "retention_correct": retention, "retention_total": 50,
            "safety_regressions": safety, "tokens_per_s": throughput}


def signed(receipt, key=KEY):
    payload = copy.deepcopy(receipt)
    payload.pop("hmac_sha256", None)
    payload["hmac_sha256"] = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    return payload


def receipt(report):
    rows = []
    for seed in report["seeds"]:
        rows.append({
            "seed": seed, "baseline_run_id": f"{seed:032x}",
            "candidate_run_id": f"{seed + 100:032x}",
            "candidate_optimizer": "adamw", "candidate_learning_rate": 9e-5,
            "starting_checkpoint_sha256": "c" * 64,
            "baseline": metrics(), "candidate": metrics(correct=39),
        })
    return signed({
        "schema": SCHEMA, "base_sha256": BASE,
        "report_sha256": observe(report)["report_sha256"],
        "train_split_sha256": "a" * 64, "selection_split_sha256": "b" * 64,
        "eval_run_id": "a" * 32,
        "source": "independent-isolated-evaluator", "evaluations": rows,
    })


def test_paired_independent_receipt_only_requires_manual_review(report):
    result = review_trial(report, model_proposal(report), receipt(report), KEY)
    assert result["status"] == "manual_review_required"
    assert result["receipt_hmac_verified"] is True
    assert len(result["comparisons"]) == 3
    assert result["provisional_regression_flag"] is False
    assert result["statistical_significance_established"] is False
    assert result["trial_authorized"] is False
    assert result["weight_promotion_permitted"] is False


def test_missing_key_blocks(report):
    result = review_trial(report, model_proposal(report), receipt(report), b"")
    assert result["reason"] == "missing_independent_evaluator_key"


def test_wrong_key_and_tampering_block(report):
    original = receipt(report)
    assert review_trial(report, model_proposal(report), original, b"z" * 32)["status"] == "blocked"
    manipulated = copy.deepcopy(original)
    manipulated["evaluations"][0]["candidate"]["task_correct"] = 50
    assert review_trial(report, model_proposal(report), manipulated, KEY)["reason"] == "receipt_authentication_failed"


@pytest.mark.parametrize("mutation, expected", [
    (lambda r, rep: r.update({"selection_split_sha256": r["train_split_sha256"]}),
     "evaluation_split_not_independent"),
    (lambda r, rep: r.update({"selection_split_sha256": rep["arms"][0]["train_sha256"]}),
     "evaluation_split_not_independent"),
    (lambda r, rep: r["evaluations"].pop(), "invalid_evaluation_inventory"),
    (lambda r, rep: r["evaluations"][1].update({"seed": r["evaluations"][0]["seed"]}),
     "unpaired_optimizer_or_seed"),
    (lambda r, rep: r["evaluations"][1].update({"baseline_run_id": r["evaluations"][0]["baseline_run_id"]}),
     "reused_evaluation_run_id"),
    (lambda r, rep: r["evaluations"][0].update({"starting_checkpoint_sha256": "invalid"}),
     "missing_starting_checkpoint_identity"),
    (lambda r, rep: r["evaluations"][0]["candidate"].update({"task_total": 43}),
     "unpaired_task_evaluation"),
    (lambda r, rep: r["evaluations"][0]["candidate"].update({"task_correct": 70}),
     "invalid_quality_or_safety_metrics"),
    (lambda r, rep: r["evaluations"][0]["candidate"].update({"tokens_per_s": float("nan")}),
     "unserializable_receipt"),
    (lambda r, rep: r["evaluations"][0].update({"candidate_learning_rate": 2e-4}),
     "unpaired_optimizer_or_seed"),
    (lambda r, rep: r.update({"source": "self_reported_by_llm"}),
     "evaluation_custody_mismatch"),
])
def test_resigned_but_invalid_receipt_still_blocked(report, mutation, expected):
    value = receipt(report)
    mutation(value, report)
    # Nonfinite values cannot be serialized into an authenticated receipt.
    if expected == "unserializable_receipt":
        value["hmac_sha256"] = "f" * 64
    else:
        value = signed(value)
    out = review_trial(report, model_proposal(report), value, KEY)
    assert out["reason"] == expected


def test_signed_regression_flags_even_when_task_mean_improves(report):
    value = receipt(report)
    value["evaluations"][0]["candidate"]["safety_regressions"] = 1
    value = signed(value)
    result = review_trial(report, model_proposal(report), value, KEY)
    assert result["status"] == "manual_review_required"
    assert result["provisional_regression_flag"] is True
    assert result["weight_promotion_permitted"] is False


def test_stale_proposal_not_accepted_even_with_signed_evaluation(report):
    wrong = model_proposal(report)
    wrong["report_sha256"] = "0" * 64
    assert review_trial(report, wrong, receipt(report), KEY)["reason"] == "unverified_model_proposal"

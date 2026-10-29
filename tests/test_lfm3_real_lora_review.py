"""Offline report contract tests against checked-in real-model research evidence."""

import copy
import json
from pathlib import Path

import pytest

from src.lfm3_real_lora_review import review

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "research/evidence/lfm3-real-lora-20261029"
    / "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def real_report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_verified_real_research_report_is_review_only(real_report):
    out = review(real_report)
    assert out["status"] == "review_required"
    assert out["weight_promotion_permitted"] is False
    assert out["proposal"]["optimizer_switch_permitted"] is False
    assert out["proposal"]["muon_on_lora_factors_validated"] is False
    assert out["seed_count"] == 3 and out["steps_per_arm"] == 20
    assert len(out["per_seed"]) == 3


def test_all_observed_seeds_have_unique_runs_and_paired_adapters(real_report):
    rows = real_report["arms"]
    assert len({a["run_id"] for a in rows}) == 6
    for i in range(0, len(rows), 2):
        a, m = rows[i : i + 2]
        assert a["seed"] == m["seed"]
        assert a["initial_adapter_sha256"] == m["initial_adapter_sha256"]
        assert a["train_sha256"] == m["train_sha256"]
        assert a["heldout_sha256"] == m["heldout_sha256"]
        assert a["heldout_loss_before"] == m["heldout_loss_before"]
        assert a["train_sha256"] != a["heldout_sha256"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda r: r.update({"weight_promotion_permitted": True}), "unverified_model"),
        (lambda r: r.update({"synthetic_only": False}), "unverified_model"),
        (lambda r: r.update({"real_private_traces_used": True}), "unverified_model"),
        (lambda r: r.update({"base_revision": "unverified"}), "unverified_model"),
        (lambda r: r["arms"][0].update({"base_sha256": "a" * 64}), "invalid_arm"),
        (lambda r: r["arms"][0].update({"frozen_backbone_sample_unchanged": False}), "invalid_arm"),
        (lambda r: r["arms"][0].update({"trainable_parameters": 123}), "invalid_arm"),
        (lambda r: r["arms"][0].update({"initial_adapter_sha256": "nonsense"}), "invalid_adapter"),
        (lambda r: r["arms"][0]["events"].pop(), "incomplete_step"),
        (lambda r: r["arms"][0]["events"][0].update({"step": 4}), "step_provenance"),
        (
            lambda r: r["arms"][0]["events"][0].update({"train_loss": float("nan")}),
            "invalid_step_metric",
        ),
        (lambda r: r["arms"][0].update({"heldout_loss_after": float("inf")}), "invalid_holdout"),
        (lambda r: r["arms"][0].update({"heldout_loss_after": -1.0}), "inconsistent_holdout"),
        (lambda r: r["arms"][0].update({"weight_promotion_permitted": True}), "invalid_arm"),
        (lambda r: r["arms"][1].update({"initial_adapter_sha256": "b" * 64}), "unpaired_start"),
        (lambda r: r["arms"][1].update({"run_id": r["arms"][0]["run_id"]}), "invalid_arm"),
        (lambda r: r["arms"].pop(), "invalid_seed_step"),
        (lambda r: r.update({"seeds": [1, 1, 3]}), "invalid_seed_step"),
        (
            lambda r: r["arms"][1]["events"][8].update({"dataset_sha256": "c" * 64}),
            "step_provenance",
        ),
        (lambda r: r["arms"][0].update({"learning_rate": 8.0}), "invalid_arm"),
    ],
)
def test_tampering_fails_closed(real_report, mutation, reason):
    modified = copy.deepcopy(real_report)
    mutation(modified)
    out = review(modified)
    assert out["status"] == "blocked"
    assert out["weight_promotion_permitted"] is False
    assert reason in out["reason"]


def test_reviewer_never_claims_measured_quality_improvement(real_report):
    result = review(real_report)
    assert result["synthetic_only"] is True
    assert result["proposal"]["action"] == "collect_new_independent_real_data_before_tuning"
    assert all(
        set(p)
        == {
            "seed",
            "adamw_heldout",
            "muon_heldout",
            "muon_minus_adamw",
        }
        for p in result["per_seed"]
    )


def test_independent_seed1_repeat_exactly_reproduces_adapter_trajectory(real_report):
    repeat_path = EVIDENCE.parent / "seed1-repeat-20step-20261029.json"
    again = json.loads(repeat_path.read_text(encoding="utf-8"))
    assert review(again)["status"] == "review_required"
    for current in again["arms"]:
        original = next(
            a
            for a in real_report["arms"]
            if a["seed"] == current["seed"] and a["optimizer"] == current["optimizer"]
        )
        assert original["run_id"] != current["run_id"]
        assert original["initial_adapter_sha256"] == current["initial_adapter_sha256"]
        assert original["final_adapter_sha256"] == current["final_adapter_sha256"]
        assert original["heldout_loss_after"] == current["heldout_loss_after"]
        assert [e["train_loss"] for e in original["events"]] == [
            e["train_loss"] for e in current["events"]
        ]
"""Fail-closed synthetic sweep review tests with actual CPU optimizer traces."""

from __future__ import annotations

import copy

import pytest

from src.lfm3_optimizer_ab_harness import benchmark
from src.lfm3_optimizer_sweep_review import review


@pytest.fixture(scope="module")
def reports():
    torch = pytest.importorskip("torch")
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("PyTorch build has no torch.optim.Muon; synthetic trace fixture unavailable")
    return [benchmark(seeds=(1, 2, 3), steps=20, muon_lr=lr) for lr in (0.0001, 0.0005, 0.001)]


def test_valid_synthetic_grid_proposes_review_only(reports):
    result = review(reports)
    assert result["status"] == "review_required"
    assert result["weight_promotion_permitted"] is False
    assert result["proposal"]["may_change_running_optimizer"] is False
    assert result["proposal"]["requires_new_independent_heldout_evaluation"]
    assert result["seed_count"] == 3
    assert len(result["results"]) == 3


def test_less_than_three_lr_regimes_not_enough(reports):
    assert review(reports[:2])["reason"] == "invalid_report_count"


def test_duplicate_muon_lr_blocks(reports):
    r = [reports[0], reports[1], reports[1]]
    assert review(r)["reason"] == "duplicate_muon_setting"


def test_data_custody_mismatch_blocks(reports):
    r = copy.deepcopy(reports)
    r[1]["arms"][1]["holdout_sha256"] = "a" * 64
    assert review(r)["reason"] == "mismatched_paired_arm"


def test_baseline_checkpoint_drift_blocks(reports):
    r = copy.deepcopy(reports)
    r[2]["arms"][0]["checkpoint_after_sha256"] = "f" * 64
    assert review(r)["reason"] == "baseline_drift_between_reports"


def test_telemetry_tamper_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"][0]["events"][5]["step"] = 100
    assert review(r)["reason"].startswith("untrusted_telemetry:")


def test_promotion_flag_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["weight_promotion_permitted"] = True
    assert review(r)["reason"] == "report_not_synthetic_or_promotion_safe"


def test_optimizer_result_claiming_promotion_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"][0]["weights_promoted"] = True
    assert review(r)["reason"] == "unsafe_or_duplicate_arm"


def test_missing_arm_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"].pop()
    assert review(r)["reason"] == "invalid_arm_inventory"


def test_mismatched_steps_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["steps_per_arm"] = 30
    assert review(r)["reason"] == "unsafe_or_duplicate_arm"


def test_differing_seed_inventory_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["seeds"][0] = 100
    assert review(r)["reason"] == "unsafe_or_duplicate_arm"


def test_inconsistent_learning_rate_metadata_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"][1]["learning_rate"] = 0.001
    assert review(r)["reason"] == "telemetry_custody_mismatch"


def test_nonfinite_loss_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"][0]["holdout_loss_after"] = float("nan")
    assert review(r)["reason"] == "invalid_holdout_metric"


def test_bias_fallback_rate_drift_blocks(reports):
    r = copy.deepcopy(reports)
    r[0]["arms"][1]["adamw_fallback_lr"] = 0.0002
    assert review(r)["reason"] == "adamw_fallback_learning_rate_drift"
"""Verifier tests for actual CPU-only, synthetic-token 3-LR LFM3 optimizer runs."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_lrgrid_review import LR_LEVELS, analyze

EVIDENCE = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-real-lora-20261029"
FILES = [
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json",
    "lfm3-real-lora-lrgrid-adamw8e-5-muon12e-5-20261029.json",
    "lfm3-real-lora-lrgrid-adamw12e-5-muon8e-5-20261029.json",
]


@pytest.fixture(scope="module")
def reports():
    return [json.loads((EVIDENCE / f).read_text(encoding="utf-8")) for f in FILES]


def test_360_real_optimizer_steps_cross_verified_but_not_promotable(reports):
    result = analyze(reports)
    assert result["status"] == "descriptive_synthetic_grid_only"
    assert result["runs_verified"] == 18
    assert result["optimizer_steps_verified"] == 360
    assert result["lr_levels"] == list(LR_LEVELS)
    assert len(result["cells"]) == 6
    assert result["real_task_quality_measured"] is False
    assert result["optimizer_or_lr_selected"] is None
    assert result["weight_promotion_permitted"] is False
    assert all(c["seed_count"] == 3 for c in result["cells"])


def test_input_order_does_not_change_grid(reports):
    assert analyze(reports)["cells"] == analyze(list(reversed(reports)))["cells"]


def test_cross_report_replayed_run_id_denied(reports):
    changed = copy.deepcopy(reports)
    changed[1]["arms"][0]["run_id"] = changed[0]["arms"][0]["run_id"]
    assert analyze(changed)["reason"] == "replayed_optimizer_run_id"


def test_cross_report_adapter_drift_denied_even_when_individually_paired(reports):
    changed = copy.deepcopy(reports)
    for arm in changed[1]["arms"]:
        if arm["seed"] == 1:
            arm["initial_adapter_sha256"] = "a" * 64
    assert analyze(changed)["reason"] == "unpaired_starting_state_between_lr_trials"


def test_model_mismatch_denied(reports):
    changed = copy.deepcopy(reports)
    changed[1]["base_sha256"] = "a" * 64
    assert analyze(changed)["status"] == "blocked"


def test_missing_or_duplicate_grid_denied(reports):
    for candidate in ([reports[0], reports[1]], [reports[0], reports[0], reports[2]]):
        assert analyze(candidate)["status"] == "blocked"


def test_nonfinite_metric_denied(reports):
    changed = copy.deepcopy(reports)
    changed[1]["arms"][0]["events"][0]["train_loss"] = float("nan")
    assert analyze(changed)["status"] == "blocked"


def test_wrong_heldout_dataset_denied(reports):
    changed = copy.deepcopy(reports)
    for arm in changed[2]["arms"]:
        if arm["seed"] == 1:
            arm["heldout_sha256"] = "d" * 64
    assert analyze(changed)["reason"] == "unpaired_starting_state_between_lr_trials"


def test_no_automatic_lr_selection_despite_apparent_loss_differences(reports):
    result = analyze(reports)
    assert result["status"] == "descriptive_synthetic_grid_only"
    assert result["optimizer_or_lr_selected"] is None
    assert result["trial_authorized"] is False
    assert result["no_statistical_superiority_claim"] is True

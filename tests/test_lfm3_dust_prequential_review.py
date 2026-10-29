"""Reject tampered or cherry-picked LFM3 forecast evidence (no model required)."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_prequential_review import review

ROOT = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-dust-prequential-20261029"


def report(seed=1):
    return json.loads((ROOT / f"seed{seed}-all-candidates-8episodes.json").read_text())


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_real_forward_only_all_candidate_evidence_structurally_valid(seed):
    result = review(report(seed))
    assert result["status"] == "descriptive_review_required"
    assert result["seed"] == seed
    assert result["prequential_cases_per_policy"] == 56
    assert result["adapter_unchanged"] is True
    assert result["independent_execution_attestation"] is False
    assert result["model_quality_improvement_proven"] is False
    assert result["optimizer_policy_promotion_permitted"] is False
    assert result["weight_promotion_permitted"] is False


@pytest.mark.parametrize("tamper", [
    lambda x: x["episode_receipts"][1]["candidates"][0]["predictions"].update({"calibrated": 9}),
    lambda x: x["episode_receipts"][1]["candidates"][0].update({"measured_gain": 9}),
    lambda x: x["episode_receipts"][0]["candidates"][0].update({"calibrated_evidence": 8}),
    lambda x: x["episode_receipts"][1]["candidates"][0].update({"direction_evidence": 12}),
    lambda x: x["episode_receipts"][2]["candidates"][3].update({"candidate": 0}),
    lambda x: x["episode_receipts"][1]["candidates"].pop(),
    lambda x: x["episode_receipts"][2]["candidates"][1].update({"sign": 1}),
    lambda x: x["episode_receipts"][1]["candidates"][0].update({"head_weights_preprobe": [1, 0]}),
    lambda x: x["episode_receipts"][1].update({"context_sha256": "0"}),
    lambda x: x["episode_receipts"][3].update({"calibration_event_count_after": 1}),
    lambda x: x["episode_receipts"][2].update({"training_backward_count": 1}),
    lambda x: x.update({"model_adapter_update_count": 1}),
    lambda x: x.update({"selection_partition_evaluated": True}),
    lambda x: x.update({"independent_test_evaluated": True}),
    lambda x: x.update({"weight_promotion_permitted": True}),
    lambda x: x.update({"source_independently_authenticated": True}),
])
def test_adversarial_replay_fails_closed(tamper):
    source = copy.deepcopy(report(1))
    tamper(source)
    assert review(source)["status"] == "blocked"


def test_renamed_seed_cannot_borrow_other_seed_ledger():
    source = report(2)
    source["seed"] = 1
    assert review(source)["status"] == "blocked"


def test_zero_mean_squared_error_is_not_improvement_claim():
    source = report(3)
    source["metrics"]["calibrated"]["mse"] = 0.0
    assert review(source)["status"] == "blocked"

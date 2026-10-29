"""Verify committed, real CPU-only Dust/LFM3 paired research receipts."""
import copy
import json
from pathlib import Path

import pytest
from src.lfm3_dust_forward_review import verify, pair

ROOT = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-dust-forward-20261029"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_real_paired_forward_only_evidence_is_structurally_valid(seed):
    attended = json.loads((ROOT / f"seed{seed}-attended-3steps.json").read_text())
    baseline = json.loads((ROOT / f"seed{seed}-random-3steps.json").read_text())
    result = pair(attended, baseline)
    assert result["status"] == "paired_manual_review_only"
    assert result["independent_improvement_established"] is False
    assert result["weight_promotion_permitted"] is False
    assert result["attended"]["source_authenticated"] is False
    assert result["attended"]["zero_backward_execution_independently_proven"] is False
    assert result["attended"]["score_after"] == (5, 7, 6)[seed - 1]
    assert result["random_control"]["score_after"] == (5, 7, 6)[seed - 1]


def test_tampered_probe_budget_denied():
    example = json.loads((ROOT / "seed1-attended-3steps.json").read_text())
    changed = copy.deepcopy(example)
    changed["records"][0]["actual_antithetic_forwards"] = 6
    assert verify(changed)["status"] == "blocked"


def test_holdout_lookahead_and_promotion_denied():
    example = json.loads((ROOT / "seed2-attended-3steps.json").read_text())
    for alteration in ({"final_test_evaluated": True},
                       {"trial_authorized": True},
                       {"weight_promotion_permitted": True},
                       {"gradient_backwards": 1}):
        changed = {**example, **alteration}
        assert verify(changed)["status"] == "blocked"


def test_mismatched_seed_rejected():
    a = json.loads((ROOT / "seed1-attended-3steps.json").read_text())
    b = json.loads((ROOT / "seed2-random-3steps.json").read_text())
    assert pair(a, b)["status"] == "blocked"

"""Adversarial and causal acceptance tests for activation-head reports."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_activation_review import (
    review_activation, compare_activation_to_control,
)

ROOT = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-dust-forward-20261029"
ACTIVATION = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-dust-activation-20261029"


@pytest.mark.parametrize("seed", [1,2,3])
def test_real_activation_head_receipts_remain_manual_only(seed):
    a = json.loads((ACTIVATION / f"seed{seed}-activation-attended-3steps.json").read_text())
    b = json.loads((ROOT / f"seed{seed}-random-3steps.json").read_text())
    record = compare_activation_to_control(a,b)
    assert record["status"] == "paired_manual_review_required"
    assert record["additional_prompt_forward_overhead"] == 3
    assert record["activation"]["trained_transformer_attention_weights"] is False
    assert record["training_improvement_or_benefit_proven"] is False
    assert record["weight_promotion_permitted"] is False


def test_tampered_context_or_extra_forward_budget_rejected():
    src = json.loads((ACTIVATION / "seed1-activation-attended-3steps.json").read_text())
    for change in (
        lambda r: r["records"][0].update({"prompt_context_sha256":"not-a-digest"}),
        lambda r: r["records"][0].update({"prompt_only_extra_forwards":0}),
        lambda r: r["records"][1].update({"attention_head_disagreement":[-1,0,0,0]}),
        lambda r: r.update({"raw_activations_persisted":True}),
        lambda r: r.update({"extra_prompt_only_forwards":0}),
        lambda r: r.update({"weight_promotion_permitted":True}),
        lambda r: r.update({"final_test_evaluated":True}),
    ):
        changed = copy.deepcopy(src)
        change(changed)
        assert review_activation(changed)["status"] == "blocked"


def test_mismatched_random_control_rejected():
    a=json.loads((ACTIVATION / "seed1-activation-attended-3steps.json").read_text())
    b=json.loads((ROOT / "seed2-random-3steps.json").read_text())
    assert compare_activation_to_control(a,b)["status"]=="blocked"

"""Strict three-seed cross-family LFM3 predictive-head review and admission."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_text_family_transfer_review import review
from src.lfm3_dust_text_transfer_admission import assess

ROOT=Path(__file__).resolve().parents[1]/"research/evidence/lfm3-dust-text-transfer-20261029"


def source(seed):
    return json.loads((ROOT/f"seed{seed}-extraction-to-uppercase-18episodes.json").read_text())


@pytest.mark.parametrize("seed",[10,11,12])
def test_real_model_neural_head_freezes_on_new_task_family(seed):
    data=source(seed)
    r=review(data)
    assert r["status"]=="manual_transfer_review_only"
    assert r["train_prompts"]==12
    assert r["transfer_prompts"]==6
    assert r["transfer_cases"]==48
    assert r["neural_head_updated"] is True
    assert r["evaluation_feedback_never_used"] is True
    assert r["model_adapter_unchanged"] is True
    assert r["optimizer_switch_permitted"] is False
    assert r["weight_promotion_permitted"] is False
    assert data["initial_adapter_sha256"]==data["final_adapter_sha256"]
    assert data["head_frozen_before_transfer_sha256"]==data["head_final_sha256"]
    assert data["source_model_perturbed_forwards"]==144
    assert data["source_model_context_forwards"]==18
    for row in data["events"][12:]:
        assert row["family"]=="uppercase"
        if row["episode"]>12:
            assert row["head_training_before_forecast"] is None
        assert row["neural_head_sha_after_episode"]==data["head_final_sha256"]


def test_transfer_does_not_establish_autonomous_optimizer_superiority():
    r=assess([source(s) for s in (10,11,12)])
    assert r["status"]=="hold_not_qualified"
    assert r["transfer_signed_direction_forecasts_per_policy"]==144
    assert r["training_examples_unique"]==36
    assert r["unseen_instruction_examples_unique"]==18
    assert r["predictor_qualified_for_model_updates"] is False
    assert r["optimizer_switch_permitted"] is False
    assert r["weight_promotion_permitted"] is False


@pytest.mark.parametrize("tamper",[
    lambda x:x["events"][13]["signed_directions"][0]["predictions"].update({"neural":10}),
    lambda x:x["events"][13]["signed_directions"][0].update({"measured_gain":10}),
    lambda x:x["events"][12]["head_training_before_forecast"].update({"head_updates_accepted":99}),
    lambda x:x["events"][14].update({"head_training_before_forecast":{"episode":14}}),
    lambda x:x["events"][15].update({"neural_head_sha_after_episode":"f"*64}),
    lambda x:x["events"][15]["context"].__setitem__(0,0.9),
    lambda x:x["events"][15]["signed_directions"].pop(),
    lambda x:x["events"][10].update({"family":"uppercase"}),
    lambda x:x["events"][5]["signed_directions"][1].update({"sign":1}),
    lambda x:x["transfer_metrics"]["neural"].update({"mean_squared_error":0.}),
    lambda x:x.update({"transfer_family_feedback_used_for_training":True}),
    lambda x:x.update({"model_adapter_update_count":1}),
    lambda x:x.update({"original_untouched_test_opened":True}),
    lambda x:x.update({"weight_promotion_permitted":True}),
])
def test_reviewer_rejects_transfer_label_leak_or_forged_gains(tamper):
    x=copy.deepcopy(source(10))
    tamper(x)
    assert review(x)["status"]=="blocked"


def test_missing_or_duplicate_seed_not_admitted():
    data=[source(s) for s in (10,11,12)]
    assert assess(data[:-1])["status"]=="blocked"
    assert assess([data[0],data[0],data[2]])["status"]=="blocked"
    altered=copy.deepcopy(data)
    altered[2]["seed"]=10
    assert assess(altered)["status"]=="blocked"

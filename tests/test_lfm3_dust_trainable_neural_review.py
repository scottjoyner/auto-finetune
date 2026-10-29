"""Frozen real LFM3 trainable cross-attention outcomes and strict no-release gates."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_trainable_neural_review import review
from src.lfm3_dust_trainable_neural_admission import assess

ROOT=Path(__file__).resolve().parents[1]/"research/evidence/lfm3-dust-trainable-head-20261029"


def source(seed):
    return json.loads((ROOT/f"seed{seed}-neural-head-12episodes.json").read_text())


@pytest.mark.parametrize("seed",[7,8,9])
def test_real_pretrained_lfm3_head_training_replays_from_only_prior_labels(seed):
    receipt=source(seed)
    r=review(receipt)
    assert r["status"]=="manual_neural_head_review_only"
    assert r["head_trainable_params"]==195
    assert r["head_updated_from_prior_forward_only_feedback"] is True
    assert r["total_head_only_loss_evaluations"]==50
    assert r["prospective_cases"]==88
    assert r["new_model_backward_calls"]==0
    assert r["model_adapter_unchanged"] is True
    assert r["head_policy_qualified"] is False
    assert r["weight_promotion_permitted"] is False
    assert receipt["neural_head_final_sha256"] == receipt["episode_receipts"][-1]["neural_head_after_episode_sha256"]
    assert receipt["total_perturbed_forwards"]==96
    assert receipt["total_context_forwards"]==12
    assert receipt["initial_adapter_sha256"]==receipt["final_adapter_sha256"]


def test_new_neural_head_does_not_pass_cross_seed_admission():
    result=assess([source(s) for s in (7,8,9)])
    assert result["status"]=="hold_not_qualified"
    assert result["head_parameters_trained_without_backprop"] is True
    assert result["every_seed_forecast_mse_5pct_below_zero"] is False
    assert result["every_seed_top1_gain_advantage_positive"] is False
    assert result["prospective_forecasts_per_method"]==264
    assert result["unique_training_task_examples"]>=32
    assert result["model_update_permitted"] is False
    assert result["weight_promotion_permitted"] is False
    assert result["per_seed"]["9"]["neural_mse_ratio_vs_zero"]<1
    assert result["per_seed"]["7"]["neural_mse_ratio_vs_zero"]>1
    assert result["per_seed"]["8"]["neural_mse_ratio_vs_zero"]>1


@pytest.mark.parametrize("tamper",[
    lambda r:r["episode_receipts"][2]["candidates"][0]["predictions"].update({"neural":100}),
    lambda r:r["episode_receipts"][2]["neural_training"].update({"head_updates_accepted":9}),
    lambda r:r["episode_receipts"][3].update({"neural_head_after_episode_sha256":"0"*64}),
    lambda r:r["episode_receipts"][1].update({"preprobe_neural_context":[1,0,0,0]}),
    lambda r:r["episode_receipts"][2]["candidates"][0].update({"neural_historical_keys":999}),
    lambda r:r["metrics"]["neural"].update({"mse":0.0}),
    lambda r:r.update({"head_parameter_training_observed":False}),
    lambda r:r.update({"total_backward_calls":1}),
    lambda r:r.update({"final_adapter_sha256":"0"*64}),
    lambda r:r.update({"independent_test_evaluated":True}),
    lambda r:r.update({"weight_promotion_permitted":True}),
    lambda r:r["episode_receipts"][2]["candidates"][7].update({"measured_gain":100}),
])
def test_forged_future_predictions_or_head_training_blocked(tamper):
    data=source(7)
    tamper(data)
    assert review(data)["status"]=="blocked"


def test_different_seeds_not_substitutable():
    data=[source(s) for s in (7,8,9)]
    assert assess([data[0],data[0],data[2]])["status"]=="blocked"
    altered=copy.deepcopy(data)
    altered[2]["seed"]=7
    assert assess(altered)["status"]=="blocked"
    assert assess(data[:2])["status"]=="blocked"

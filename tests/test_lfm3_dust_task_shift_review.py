"""Real pinned LFM3 new-task novelty evidence rejects false actuator claims."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_task_shift_review import review
from src.lfm3_dust_task_shift_admission import assess

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"research/evidence/lfm3-dust-task-shift-20261030"


def report(seed):
    return json.loads((EVIDENCE/f"seed{seed}-sort-two-words-shadow.json").read_text())


@pytest.mark.parametrize("seed,ood",[(13,6),(14,6),(15,5)])
def test_actual_frozen_lfm3_contexts_replay_and_abstain(seed,ood):
    r=review(report(seed),repo_root=ROOT)
    assert r["status"]=="manual_novelty_shadow_review_only"
    assert r["new_task_seed"]==seed
    assert r["new_family_ood_episode_count"]==ood
    assert r["candidate_abstentions"]==48
    assert r["new_family_examples"]==6
    assert r["prompt_only_model_forwards"]==6
    assert r["model_adapter_modified"] is False
    assert r["head_update_authorized"] is False
    assert r["actual_model_probe_gains_evaluated"] is False
    assert r["optimizer_switch_permitted"] is False
    assert r["weight_promotion_permitted"] is False


def test_all_three_cohorts_verified_no_hidden_training_or_forecast_mse():
    a=assess([report(s) for s in (13,14,15)],repo_root=ROOT)
    assert a["status"]=="shadow_guard_validated_not_optimizer_qualified"
    assert a["novel_task_contexts_flagged"]==17
    assert a["novel_task_contexts_not_flagged"]==1
    assert a["prompt_only_real_pretrained_model_forwards"]==18
    assert a["candidate_predictions_abstained"]==144
    assert a["candidate_predictions_total"]==144
    assert a["measured_new_family_candidate_gains"]==0
    assert a["measured_new_family_predictive_mse"] is None
    assert a["saved_new_family_perturbation_forwards"] is None
    assert all(x is False for x in a["head_training_only_gain_skill_per_seed"].values())
    assert a["optimizer_switch_permitted"] is False
    assert a["model_update_permitted"] is False
    assert a["weight_promotion_permitted"] is False


@pytest.mark.parametrize("change",[
    lambda j:j["episodes"][0]["raw_head_predicted_directions"][0]["guard"].update({"abstained":False}),
    lambda j:j["episodes"][0]["raw_head_predicted_directions"][0]["guard"].update({"in_training_context_envelope":True}),
    lambda j:j["episodes"][0]["raw_head_predicted_directions"][0].update({"measured_gain":.1}),
    lambda j:j["episodes"][0]["raw_head_predicted_directions"][0].update({"actual_perturbation_forward_performed":True}),
    lambda j:j["episodes"][1]["raw_head_predicted_directions"].pop(),
    lambda j:j["episodes"][2]["raw_head_predicted_directions"][0].update({"raw_neural_prediction":20.}),
    lambda j:j["episodes"][3]["raw_head_predicted_directions"][1]["direction"].__setitem__(0,.6),
    lambda j:j["episodes"][4].update({"head_parameter_updates":1}),
    lambda j:j["episodes"][5].update({"context":[1.,0.,0.,0.]}),
    lambda j:j.update({"source_model_perturbation_forwards":8}),
    lambda j:j.update({"source_training_receipt_sha256":"f"*64}),
    lambda j:j.update({"model_training_permitted":True}),
    lambda j:j.update({"weight_promotion_permitted":True}),
])
def test_old_training_math_and_new_preprobe_events_fail_closed(change):
    x=copy.deepcopy(report(13))
    change(x)
    result=review(x,repo_root=ROOT)
    assert result["status"]=="blocked"
    assert result["optimizer_switch_permitted"] is False
    assert result["weight_promotion_permitted"] is False


def test_duplicate_or_partial_cohort_is_not_admitted():
    reports=[report(s) for s in (13,14,15)]
    assert assess(reports[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([reports[0],reports[0],reports[2]],repo_root=ROOT)["status"]=="blocked"
    x=copy.deepcopy(reports)
    x[2]["new_task_seed"]=13
    assert assess(x,repo_root=ROOT)["status"]=="blocked"


def test_source_receipt_is_fixed_and_cannot_be_substituted(tmp_path):
    x=report(13)
    assert review(x,repo_root=tmp_path)["status"]=="blocked"

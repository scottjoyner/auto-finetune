"""Real LFM3 three-episode online 195-weight head: causal evidence and HOLD."""
import copy
import json
from pathlib import Path
import pytest

from src.lfm3_dust_online_one_feedback_review import review
from src.lfm3_dust_online_one_feedback_admission import assess

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"research/evidence/lfm3-dust-online-one-feedback-20261030"


def source(seed):
    return json.loads((DATA/f"seed{seed}-three-episodes.json").read_text())


@pytest.mark.parametrize("seed,head_forwards",[(28,7),(29,7),(30,4)])
def test_real_backprop_free_195_qk_weights_adapt_only_to_previous_one_probe(seed,head_forwards):
    x=source(seed)
    r=review(x,repo_root=ROOT)
    assert r["status"]=="manual_causal_one_feedback_head_review_only"
    assert r["head_qk_weights_learn_from_prior_single_train_gains"] is True
    assert r["total_online_head_only_objectives"]==head_forwards
    assert r["total_online_head_weight_updates"]==2
    assert r["real_pretrained_model_loss_forwards"]==84
    assert r["real_model_prompt_context_forwards"]==3
    assert r["frozen_head_unchanged"] is True
    assert r["same_2048_model_lora_scope"] is True
    assert r["model_persistent_adapter_updates"]==0
    assert x["model_adapter_initial_sha256"]==x["model_adapter_final_sha256"]
    assert x["online_start_head_sha256"]!=x["online_end_head_sha256"]
    assert x["online_head_parameters"]==195
    assert len(r["episodes"])==3
    assert all(e["policies"]["adaptive_head_one_probe"]["selected_index"]==
               e["policies"]["frozen_head_one_probe"]["selected_index"]
               for e in r["episodes"])
    assert not r["model_training_permitted"]
    assert not r["weight_promotion_permitted"]


def test_online_parameter_change_does_not_imply_training_quality_gain():
    r=assess([source(s) for s in (28,29,30)],repo_root=ROOT)
    assert r["status"]=="hold_not_qualified"
    assert r["actual_accepted_online_qk_head_updates"]==6
    assert r["actual_online_head_only_objective_forwards"]==18
    assert r["actual_model_loss_forwards"]==252
    assert r["actual_model_prompt_context_forwards"]==9
    assert r["episode_choices_changed_by_online_learning"]==0
    assert r["total_policy_episodes"]==9
    assert r["training_feedback_labels_per_episode"]==1
    assert r["actual_head_weights_changed_without_model_backprop"]
    assert r["model_update_effectiveness_improved_proven"] is False
    assert r["persistent_model_adapter_updates"]==0
    assert r["policy_outcomes"]["adaptive_head_one_probe"][
        "mean_episode_diagnostic_loss_delta"]==r["policy_outcomes"][
            "frozen_head_one_probe"]["mean_episode_diagnostic_loss_delta"]
    assert r["policy_outcomes"]["adaptive_head_one_probe"][
        "mean_episode_diagnostic_loss_delta"]>0
    assert not r["optimizer_switch_permitted"]
    assert not r["model_training_permitted"]
    assert not r["weight_promotion_permitted"]


@pytest.mark.parametrize("tamper",[
    lambda r:r["episodes"][0]["adaptive_forecasts_before_any_probe"].__setitem__(0,42.),
    lambda r:r["episodes"][1].update({"adaptive_head_sha_after_training_only":"f"*64}),
    lambda r:r["episodes"][1]["training_only_head_update_before_current_forecasts"].update({"head_only_accepted_updates":0}),
    lambda r:r["episodes"][1]["candidate_policy_choice"].update({"adaptive_head_one_probe":0}),
    lambda r:r["episodes"][0]["policy_results"]["adaptive_head_one_probe"].update({"train_gain":40.}),
    lambda r:r["episodes"][0]["policy_results"]["adaptive_head_one_probe"]["one_train_only_observation"].update({"gain":40.}),
    lambda r:r["episodes"][0]["policy_results"]["frozen_head_one_probe"].update({"one_train_only_observation":{"gain":.1}}),
    lambda r:r["episodes"][2]["policy_results"]["random_one_probe"].update({"model_probe_forwards":2}),
    lambda r:r["episodes"][2]["policy_results"]["adaptive_head_one_probe"].update({"mean_diagnostic_loss_delta":-99.}),
    lambda r:r["episodes"][1]["preprobe_context"].__setitem__(0,0.),
    lambda r:r["episodes"][1].update({"model_loss_forwards":27}),
    lambda r:r.update({"online_end_head_sha256":"f"*64}),
    lambda r:r.update({"source_head_receipt_sha256":"f"*64}),
    lambda r:r.update({"total_model_completion_loss_forwards":30}),
    lambda r:r.update({"model_adapter_final_sha256":"0"*64}),
    lambda r:r.update({"persistent_model_adapter_updates":1}),
    lambda r:r.update({"training_model_backward_calls":1}),
    lambda r:r.update({"trained_head_uses_diagnostic_feedback":True}),
    lambda r:r.update({"model_update_permitted":True}),
    lambda r:r.update({"weight_promotion_permitted":True}),
])
def test_head_feedback_lookahead_or_model_promotion_fails_closed(tamper):
    x=copy.deepcopy(source(28))
    tamper(x)
    assert review(x,repo_root=ROOT)["status"]=="blocked"


def test_incomplete_duplicate_or_misrouted_cohorts_rejected(tmp_path):
    data=[source(s) for s in (28,29,30)]
    assert assess(data[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([data[0],data[0],data[2]],repo_root=ROOT)["status"]=="blocked"
    y=copy.deepcopy(data)
    y[1]["seed"]=28
    assert assess(y,repo_root=ROOT)["status"]=="blocked"
    assert review(data[0],repo_root=tmp_path)["status"]=="blocked"

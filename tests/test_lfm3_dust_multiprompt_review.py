"""Actual train-side verification on real LFM3: negative multi-prompt result."""
import copy
import json
from pathlib import Path
import pytest

from src.lfm3_dust_multiprompt_review import review
from src.lfm3_dust_multiprompt_admission import assess

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"research/evidence/lfm3-dust-multiprompt-20261030"


def source(seed):
    return json.loads((EVIDENCE/f"seed{seed}-three-train-prompt-quality.json").read_text())


@pytest.mark.parametrize("seed,forwards",[(19,17),(20,25),(21,25)])
def test_replay_actual_pretrained_model_multi_prompt_diagnostics(seed,forwards):
    v=review(source(seed),repo_root=ROOT)
    assert v["status"]=="manual_multiprompt_quality_review_only"
    assert v["candidate_head_unchanged"] is True
    assert v["selected_model_scope_2048"] is True
    assert v["head_admission_failed"] is True
    assert v["total_real_model_loss_forwards"]==forwards
    assert v["optimizer_switch_permitted"] is False
    assert v["weight_promotion_permitted"] is False
    assert source(seed)["initial_adapter_sha256"]==source(seed)["final_adapter_sha256"]
    for method in ("neural","random","greedy"):
        assert v["results"][f"{method}_single"]["candidate"]==v["results"][f"{method}_multi"]["candidate"]


def test_actual_multi_prompt_guard_blocks_useful_and_admits_harmful():
    result=assess([source(s) for s in (19,20,21)],repo_root=ROOT)
    assert result["status"]=="hold_not_qualified"
    assert result["total_real_model_loss_forwards"]==67
    assert result["total_model_backwards"]==result["persistent_adapter_updates"]==0
    assert result["beneficial_single_prompt_updates_vetoed"]["greedy"]==1
    assert result["harmful_multi_prompt_updates"]["neural"]==1
    assert result["harmful_multi_prompt_updates"]["greedy"]==1
    assert result["policy_diagnostic_deltas"]["greedy"]["per_seed_single"]["20"]<0
    assert result["policy_diagnostic_deltas"]["greedy"]["per_seed_multi"]["20"]==0
    assert result["policy_diagnostic_deltas"]["neural"]["per_seed_multi"]["21"]>0
    assert not result["optimizer_switch_permitted"]
    assert not result["model_update_permitted"]
    assert not result["weight_promotion_permitted"]


@pytest.mark.parametrize("modify",[
    lambda d:d["proposals"][0].update({"predicted_gain_before_any_measurement":42.}),
    lambda d:d["proposals"][6].update({"measured_first_train_gain":-88.}),
    lambda d:d["proposals"][3]["direction"].__setitem__(0,.9),
    lambda d:d["candidate_confirmation_gains"]["6"].__setitem__(0,-1.),
    lambda d:d["results"]["neural_multi"].update({"diagnostic_mean_loss_delta":-1.}),
    lambda d:d["results"]["neural_multi"].update({"would_apply_under_own_rule":False}),
    lambda d:d["results"]["neural_multi"].update({"candidate_index":2}),
    lambda d:d["positive_first_gain_candidate_diagnostics"]["6"].update({"mean_completion_loss":0.}),
    lambda d:d.update({"model_gradients_and_backprop_calls":1}),
    lambda d:d.update({"persistent_model_adapter_updates":1}),
    lambda d:d.update({"final_adapter_sha256":"f"*64}),
    lambda d:d.update({"selection_test_partition_opened":True}),
    lambda d:d.update({"model_update_permitted":True}),
    lambda d:d.update({"weight_promotion_permitted":True}),
])
def test_modified_head_gains_training_check_or_promotions_rejected(modify):
    d=copy.deepcopy(source(21))
    modify(d)
    r=review(d,repo_root=ROOT)
    assert r["status"]=="blocked"
    assert not r["model_update_permitted"]


def test_incomplete_or_substituted_seed_blocks_new_research_admission():
    reports=[source(s) for s in (19,20,21)]
    assert assess(reports[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([reports[0],reports[0],reports[2]],repo_root=ROOT)["status"]=="blocked"
    bogus=copy.deepcopy(reports)
    bogus[0]["seed"]=21
    assert assess(bogus,repo_root=ROOT)["status"]=="blocked"

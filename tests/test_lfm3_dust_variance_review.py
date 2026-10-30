"""Real-model forward-only head with variance-aware one-step diagnostic controls."""
import copy
import json
from pathlib import Path
import pytest

from src.lfm3_dust_variance_review import review
from src.lfm3_dust_variance_admission import assess

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"research/evidence/lfm3-dust-variance-20261030"

def receipt(seed):
    return json.loads((FOLDER/f"seed{seed}-variance-gated-quality.json").read_text())

@pytest.mark.parametrize("seed,forwards",[(22,33),(23,17),(24,33)])
def test_real_pinned_lfm3_variance_gated_evidence_is_replayable(seed,forwards):
    source=receipt(seed)
    r=review(source,repo_root=ROOT)
    assert r["status"]=="manual_multiprompt_quality_review_only"
    assert r["seed"]==seed
    assert r["candidate_head_unchanged"] is True
    assert r["head_admission_failed"] is True
    assert r["selected_model_scope_2048"] is True
    assert r["variance_guard_is_heuristic_not_confidence_interval"] is True
    assert r["total_real_model_loss_forwards"]==forwards
    assert r["model_update_permitted"] is False
    assert r["weight_promotion_permitted"] is False
    assert source["initial_adapter_sha256"]==source["final_adapter_sha256"]
    for policy in ("random","neural","greedy"):
        assert r["results"][f"{policy}_single"]["candidate"]==r["results"][f"{policy}_variance"]["candidate"]

def test_held_out_results_show_variance_false_veto_and_some_real_neural_gain():
    a=assess([receipt(s) for s in (22,23,24)],repo_root=ROOT)
    assert a["status"]=="hold_not_qualified"
    assert a["real_model_loss_forwards"]==83
    assert a["real_prompt_activation_forwards"]==3
    assert a["model_backpropagation_calls"]==0
    assert a["persistent_adapter_updates"]==0
    assert a["methods"]["neural"]["useful_single_updates_vetoed_by_variance"]==1
    assert a["methods"]["greedy"]["useful_single_updates_vetoed_by_variance"]==1
    assert a["methods"]["neural"]["variance"]["temporary_steps_applied"]==1
    assert a["methods"]["neural"]["variance"]["per_seed_diagnostic_loss_delta"]["24"]<0
    assert a["methods"]["neural"]["single"]["per_seed_diagnostic_loss_delta"]["22"]<0
    assert a["methods"]["neural"]["variance"]["per_seed_diagnostic_loss_delta"]["22"]==0
    assert a["methods"]["random"]["variance"]["per_seed_diagnostic_loss_delta"]["24"]<0
    assert not a["optimizer_switch_permitted"]
    assert not a["weight_promotion_permitted"]

@pytest.mark.parametrize("modify",[
    lambda x:x["proposals"][1].update({"predicted_gain_before_any_measurement":50}),
    lambda x:x["proposals"][6].update({"measured_first_train_gain":-50}),
    lambda x:x["proposals"][0]["direction"].__setitem__(0,0.2),
    lambda x:x["candidate_confirmation_gains"]["6"].__setitem__(0,40),
    lambda x:x["results"]["neural_variance"].update({"diagnostic_mean_loss_delta":-200}),
    lambda x:x["results"]["neural_variance"].update({"would_apply_under_own_rule":False}),
    lambda x:x["results"]["neural_variance"].update({"candidate_index":1}),
    lambda x:x["positive_first_gain_candidate_diagnostics"]["6"].update({"mean_completion_loss":0}),
    lambda x:x.update({"final_adapter_sha256":"0"*64}),
    lambda x:x.update({"persistent_model_adapter_updates":1}),
    lambda x:x.update({"model_gradients_and_backprop_calls":1}),
    lambda x:x.update({"diagnostic_labels_used_for_head_or_gate":True}),
    lambda x:x.update({"model_update_permitted":True}),
    lambda x:x.update({"weight_promotion_permitted":True}),
])
def test_any_selected_head_training_or_diagnostic_tamper_blocks(modify):
    x=copy.deepcopy(receipt(24))
    modify(x)
    assert review(x,repo_root=ROOT)["status"]=="blocked"

def test_incomplete_duplicate_or_wrong_seed_never_accepted():
    data=[receipt(s) for s in (22,23,24)]
    assert assess(data[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([data[0],data[0],data[2]],repo_root=ROOT)["status"]=="blocked"
    fake=copy.deepcopy(data)
    fake[2]["seed"]=22
    assert assess(fake,repo_root=ROOT)["status"]=="blocked"

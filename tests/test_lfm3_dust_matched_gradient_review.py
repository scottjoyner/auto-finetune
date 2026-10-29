"""Provenance and no-action checks for committed matched LFM3 optimizer controls."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_matched_gradient_review import verify,matched_pair

ROOT=Path(__file__).resolve().parents[1]/"research/evidence"
CONTROL=ROOT/"lfm3-dust-matched-three-way-20261029"
DUST=ROOT/"lfm3-dust-forward-20261029"


def inputs(seed):
    a=json.loads((CONTROL/f"seed{seed}-adamw-matched-b3.json").read_text())
    m=json.loads((CONTROL/f"seed{seed}-muon-matched-b3.json").read_text())
    d=json.loads((DUST/f"seed{seed}-random-3steps.json").read_text())
    return a,m,d


@pytest.mark.parametrize("seed", [1,2,3])
def test_real_records_reviewed_and_parameter_scope_matched(seed):
    a,m,d=inputs(seed)
    result=matched_pair(a,m,d)
    assert result["status"]=="parameter_scope_matched_manual_review"
    assert result["equal_parameter_scope"] is True
    assert result["equal_compute"] is False
    assert result["optimizer_superiority_proven"] is False
    assert result["weight_promotion_permitted"] is False
    assert result["adamw"] == result["muon"] == result["dust"] == (5,7,6)[seed-1]
    assert verify(a)["backward_calls"]==verify(m)["backward_calls"]==3
    assert a["initial_lora_b_sha256"]==m["initial_lora_b_sha256"]==d["initial_lora_b_sha256"]
    assert a["data_manifest"]==m["data_manifest"]==d["data_manifest"]


@pytest.mark.parametrize("change",[
    lambda a: a.update({"weight_promotion_permitted":True}),
    lambda a: a.update({"equal_total_forward_or_walltime_budget_with_dust":True}),
    lambda a: a.update({"selected_parameter_count":146432}),
    lambda a: a.update({"training_forwards":100}),
    lambda a: a.update({"selection_score_forwards":0}),
    lambda a: a["records"][0].update({"learning_rate":0.1}),
    lambda a: a["records"][0].update({"cumulative_lora_b_update_norm":0.2}),
    lambda a: a["records"][0].update({"raw_grad_norm":float("nan")}),
    lambda a: a["records"][1].update({"step":0}),
])
def test_explicit_provenance_and_admission_tampering_denied(change):
    a,m,d=inputs(1)
    change(a)
    assert verify(a)["status"]=="blocked"
    assert matched_pair(a,m,d)["status"]=="blocked"


def test_unpaired_seed_or_adapter_identity_denied():
    a,m,d=inputs(2)
    different=inputs(1)[1]
    assert matched_pair(a,different,d)["status"]=="blocked"
    bad=copy.deepcopy(m)
    bad["initial_lora_b_sha256"]="f"*64
    assert matched_pair(a,bad,d)["status"]=="blocked"


def test_forged_dust_forward_count_or_name_cannot_pass_three_way_review():
    a,m,d=inputs(1)
    changed=copy.deepcopy(d)
    changed["records"][0]["actual_antithetic_forwards"]=99
    assert matched_pair(a,m,changed)["status"]=="blocked"
    changed=copy.deepcopy(d)
    changed["selected_adapter_names"]=[x.replace("v_proj","q_proj")
                                       for x in changed["selected_adapter_names"]]
    assert matched_pair(a,m,changed)["status"]=="blocked"

"""Real pretrained LFM3 matched one-probe neural head versus random evidence."""
import copy
import json
from pathlib import Path
import pytest

from src.lfm3_dust_equal_probe_review import review
from src.lfm3_dust_equal_probe_admission import assess

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"research/evidence/lfm3-dust-equal-probe-20261030"


def source(seed):
    return json.loads((EVIDENCE/f"seed{seed}-matched-one-probe.json").read_text())


@pytest.mark.parametrize("seed", (25,26,27))
def test_one_actual_probe_each_and_frozen_head_replay(seed):
    v=review(source(seed),repo_root=ROOT)
    assert v["status"]=="manual_matched_probe_budget_review_only"
    assert v["selected_lora_parameters"]==2048
    assert v["model_probe_forwards_each_policy"]==1
    assert v["diagnostic_forwards_each_policy"]==6
    assert v["shared_baseline_loss_forwards"]==7
    assert v["actual_training_model_loss_forwards"]==21
    assert v["model_prompt_context_forwards_neural"]==1
    assert v["head_admission_failed"] is True
    assert v["model_weights_and_external_head_unchanged"] is True
    assert v["model_update_permitted"] is False
    assert v["weight_promotion_permitted"] is False
    raw=source(seed)
    assert raw["initial_adapter_sha256"]==raw["final_adapter_sha256"]
    assert set(v["results"])=={"random_one_probe","neural_one_probe"}
    for policy in v["results"]:
        assert 0<=v["results"][policy]["candidate_index"]<8


def test_post_update_means_and_held_out_transfer_regression_block_promotion():
    result=assess([source(s) for s in (25,26,27)],repo_root=ROOT)
    assert result["status"]=="hold_not_qualified"
    assert result["total_actual_model_completion_loss_forwards"]==63
    assert result["total_model_candidate_probes_neural"]==3
    assert result["total_model_candidate_probes_random"]==3
    assert result["total_model_prompt_only_attention_context_forwards"]==3
    assert result["model_candidate_probe_budgets_equal"] is True
    assert result["total_walltime_compute_resource_matched"] is False
    assert result["model_weights_changed_persistently"] is False
    assert result["model_training_permitted"] is False
    assert result["optimizer_switch_permitted"] is False
    assert result["weight_promotion_permitted"] is False
    neural=result["policy_outcomes"]["neural_one_probe"]
    random=result["policy_outcomes"]["random_one_probe"]
    assert neural["one_step_updates_temporarily_evaluated"]==2
    assert random["one_step_updates_temporarily_evaluated"]==0
    assert neural["per_seed_diagnostic_loss_delta"]["25"]==0
    assert neural["per_seed_diagnostic_loss_delta"]["26"]<0
    assert neural["per_seed_diagnostic_loss_delta"]["27"]>0
    assert neural["per_seed_transfer_loss_delta"]["27"]>0
    assert "unseen_uppercase_task_loss_increased" in result["blocking_reasons"]


@pytest.mark.parametrize("tamper",[
    lambda j:j["all_preprobe_neural_gain_forecasts"].__setitem__(0,100.),
    lambda j:j["candidate_directions"][2]["direction"].__setitem__(0,.9),
    lambda j:j["selected_indices_from_prehistory"].update({"neural_one_probe":7}),
    lambda j:j["policy_results"]["neural_one_probe"].update({"selected_signed_candidate":7}),
    lambda j:j["policy_results"]["neural_one_probe"].update({"measured_training_gain":100.}),
    lambda j:j["policy_results"]["neural_one_probe"].update({"diagnostic_completion_loss_delta":-100.}),
    lambda j:j["policy_results"]["random_one_probe"].update({"model_probe_forwards":8}),
    lambda j:j["policy_results"]["neural_one_probe"].update({"adapter_restored_sha256":"f"*64}),
    lambda j:j["policy_results"]["neural_one_probe"].update({"positive_gain_acceptance":False}),
    lambda j:j["preprobe_context"].__setitem__(0,0.),
    lambda j:j.update({"original_head_receipt_sha256":"a"*64}),
    lambda j:j.update({"persistent_lora_update_count":1}),
    lambda j:j.update({"neural_extra_prompt_context_forwards":0}),
    lambda j:j.update({"total_loss_model_forwards":14}),
    lambda j:j.update({"model_training_permitted":True}),
    lambda j:j.update({"weight_promotion_permitted":True}),
])
def test_forged_direction_gain_quality_or_equal_budget_is_rejected(tamper):
    j=copy.deepcopy(source(27))
    tamper(j)
    x=review(j,repo_root=ROOT)
    assert x["status"]=="blocked"
    assert not x["optimizer_switch_permitted"]
    assert not x["weight_promotion_permitted"]


def test_missing_or_duplicate_or_misrouted_source_fails_closed(tmp_path):
    data=[source(s) for s in (25,26,27)]
    assert assess(data[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([data[0],data[0],data[2]],repo_root=ROOT)["status"]=="blocked"
    forged=copy.deepcopy(data)
    forged[2]["seed"]=25
    assert assess(forged,repo_root=ROOT)["status"]=="blocked"
    assert review(data[0],repo_root=tmp_path)["status"]=="blocked"

"""Three-method LFM3 strategy interface must never bypass failed Dust trial gate."""
from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import pytest

from src.lfm3_three_method_shadow_registry import (
    METHODS,METHOD_SPEC,evidence_paths,review,shadow_rank,
)

ROOT=Path(__file__).resolve().parents[1]


def _copy_evidence(tmp_path):
    for relative in evidence_paths().values():
        source=ROOT/relative
        target=tmp_path/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    return tmp_path


def _modify(root,relative,mutator):
    p=root/relative
    data=json.loads(p.read_text())
    mutator(data)
    p.write_text(json.dumps(data,sort_keys=True,allow_nan=False))


def test_all_three_methods_have_same_exact_lora_b_scope_and_distinct_update_rules():
    r=review(ROOT)
    assert r["status"]=="verified_shadow_only"
    assert tuple(r["methods"])==METHODS==("adamw","muon","predictive_dust")
    assert r["model_adapter_parameters"]==2048
    assert r["matched_parameter_scope"] is True
    assert r["matched_total_compute"] is False
    assert r["historical_three_seed_selection_accuracy"]=={"1":5,"2":7,"3":6}
    assert len(r["evidence_sha256"])==30
    assert r["source_receipts_independently_authenticated"] is False
    for name in METHODS:
        entry=r["methods"][name]
        assert entry["name"]==name
        assert entry["trainable_lora_parameters"]==2048
        assert entry["previous_experiment_selection_scores"]=={"1":5,"2":7,"3":6}
        assert entry["implementation_executable_by_this_registry"] is False
        assert entry["model_training_permitted"] is False
        assert entry["optimizer_switch_permitted"] is False
        assert entry["weight_promotion_permitted"] is False
    assert r["methods"]["adamw"]["uses_model_backpropagation"] is True
    assert r["methods"]["muon"]["uses_model_backpropagation"] is True
    assert r["methods"]["predictive_dust"]["uses_model_backpropagation"] is False
    assert r["methods"]["predictive_dust"]["external_neural_head_parameters"]==195


def test_real_transfer_failure_blocks_predictive_method():
    r=review(ROOT,requested_method="predictive_dust")
    assert r["status"]=="verified_shadow_only"
    assert list(r["methods"])==["predictive_dust"]
    assert r["neural_head_transfer_admission"]=="hold_not_qualified"
    assert all(ratio>1 for ratio in r["neural_head_transfer_seedwise_mse_ratio"].values())
    assert r["neural_head_transfer_blockers"]
    assert r["neural_head_task_shift_shadow_admission"]=="shadow_guard_validated_not_optimizer_qualified"
    assert r["neural_head_novel_prompts_flagged"]==17
    assert r["neural_head_novel_prompts_total"]==18
    assert r["neural_head_new_task_candidates_abstained"]==144
    assert r["neural_head_one_step_quality_admission"]=="hold_not_qualified"
    assert r["one_step_persistent_model_adapter_updates"]==0
    assert r["multiprompt_quality_admission"]=="hold_not_qualified"
    assert r["multiprompt_harmful_neural_updates"]==1
    assert r["multiprompt_beneficial_greedy_updates_vetoed"]==1
    assert r["multiprompt_actual_model_loss_forwards"]==67
    assert r["variance_quality_admission"]=="hold_not_qualified"
    assert r["variance_useful_head_updates_vetoed"]==1
    assert r["variance_real_model_loss_forwards"]==83
    assert r["variance_accepted_neural_updates"]==1
    assert r["equal_one_probe_admission"]=="hold_not_qualified"
    assert r["equal_probe_model_loss_forwards"]==63
    assert r["equal_probe_neural_diagnostic_losses"]["26"]<0
    assert r["equal_probe_neural_diagnostic_losses"]["27"]>0
    assert r["equal_probe_neural_transfer_losses"]["27"]>0
    assert r["online_one_feedback_admission"]=="hold_not_qualified"
    assert r["online_one_feedback_head_updates"]==6
    assert r["online_one_feedback_model_choices_changed"]==0
    assert r["online_one_feedback_model_loss_forwards"]==252
    assert r["neural_head_one_step_diagnostic_loss_deltas"]["18"]>0
    assert r["methods"]["predictive_dust"]["head_recommendations_enabled"] is False
    assert r["model_training_permitted"] is False
    assert r["production_execution_permitted"] is False


@pytest.mark.parametrize("method",["all","adamw","muon","predictive_dust"])
def test_method_filter_is_pure_review(method):
    r=review(ROOT,requested_method=method,intent="inspect")
    assert r["status"]=="verified_shadow_only"
    assert r["trials_can_be_executed_by_this_interface"] is False
    assert r["weight_promotion_permitted"] is False
    assert len(r["methods"])==(3 if method=="all" else 1)


@pytest.mark.parametrize("method,intent",[
    ("auto","inspect"),("undefined","inspect"),("dust","inspect"),
    ("predictive_dust","train"),("adamw","train"),("muon","execute"),
    ("all","promote"),("all","merge"),("predictive_dust","auto"),
])
def test_no_automatic_optimizer_selection_or_training(method,intent):
    assert review(ROOT,requested_method=method,intent=intent)["status"]=="blocked"


@pytest.mark.parametrize("source,change",[
    ("matched_seed1_adamw",lambda x:x.update({"selected_parameter_count":4000})),
    ("matched_seed1_muon",lambda x:x.update({"actual_backwards":0})),
    ("matched_seed2_dust",lambda x:x["records"][0].update({"actual_antithetic_forwards":99})),
    ("transfer_seed10",lambda x:x["transfer_metrics"]["neural"].update({"mean_squared_error":0})),
    ("transfer_seed11",lambda x:x.update({"weight_promotion_permitted":True})),
    ("novelty_seed14",lambda x:x["episodes"][0]["raw_head_predicted_directions"][0]["guard"].update({"abstained":False})),
    ("one_step_seed18",lambda x:x["policy_results"]["neural_one_probe"].update({"diagnostic_loss_delta_after_minus_before":-100.})),
    ("three_prompt_seed21",lambda x:x["results"]["neural_multi"].update({"diagnostic_mean_loss_delta":-100.})),
    ("variance_seed24",lambda x:x["results"]["neural_variance"].update({"diagnostic_mean_loss_delta":-100.})),
    ("equal_probe_seed27",lambda x:x["policy_results"]["neural_one_probe"].update({"diagnostic_completion_loss_delta":-100.})),
    ("online_head_seed30",lambda x:x["episodes"][2]["policy_results"]["adaptive_head_one_probe"].update({"mean_diagnostic_loss_delta":-100.})),
])
def test_source_tampering_never_allows_strategy_registration(tmp_path,source,change):
    root=_copy_evidence(tmp_path)
    _modify(root,evidence_paths()[source],change)
    r=review(root)
    assert r["status"]=="blocked"
    assert r["optimizer_switch_permitted"] is False
    assert r["weight_promotion_permitted"] is False


def test_missing_provenance_or_symlink_is_not_treated_as_no_experiments(tmp_path):
    root=_copy_evidence(tmp_path)
    path=root/evidence_paths()["transfer_seed12"]
    path.unlink()
    assert review(root)["status"]=="blocked"
    target=ROOT/evidence_paths()["transfer_seed12"]
    path.symlink_to(target)
    assert review(root)["status"]=="blocked"
    assert review(tmp_path/"no-such-folder")["status"]=="blocked"


class FakeHead:
    def predict(self,coords,context,*,episode):
        return {"predicted_gain":float(coords[0])}


def test_valid_head_forecast_only_ranks_causally_and_abstains():
    vectors=[[float(i-4)]+[0.]*7 for i in range(8)]
    admission={"status":"hold_not_qualified","predictor_qualified_for_model_updates":False}
    x=shadow_rank(FakeHead(),[1.,0.,0.,0.],vectors,episode=10,admission=admission)
    assert x["status"]=="predictive_head_shadow_abstention"
    assert x["candidate_order_high_to_low"]==list(reversed(range(8)))
    assert all(c["would_actuate"] is False for c in x["candidates"])
    assert x["selected_for_live_probe"] is None
    assert x["optimizer_switch_permitted"] is False
    assert x["weight_promotion_permitted"] is False


def test_stale_or_falsely_admitted_head_cannot_actuate():
    vectors=[[0.]*8 for _ in range(8)]
    admission={"status":"eligible_for_separate_study_only",
               "predictor_qualified_for_model_updates":False}
    assert shadow_rank(FakeHead(),None,vectors,episode=1,admission=admission)["status"]=="blocked"
    admission={"status":"hold_not_qualified",
               "predictor_qualified_for_model_updates":True}
    assert shadow_rank(FakeHead(),None,vectors,episode=1,admission=admission)["status"]=="blocked"
    admission["predictor_qualified_for_model_updates"]=False
    assert shadow_rank(FakeHead(),None,vectors[:-1],episode=1,admission=admission)["status"]=="blocked"

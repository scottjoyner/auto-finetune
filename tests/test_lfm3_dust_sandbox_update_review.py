"""Rebuild frozen head and three real, in-memory one-step LoRA comparisons."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_sandbox_update_review import review
from src.lfm3_dust_sandbox_update_admission import assess

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"research/evidence/lfm3-dust-one-step-quality-20261030"


def source(seed):
    return json.loads((DATA/f"seed{seed}-one-step-diagnostic.json").read_text())


@pytest.mark.parametrize("seed",[16,17,18])
def test_exact_model_lora_quality_source_replays_without_promotions(seed):
    r=review(source(seed),repo_root=ROOT)
    assert r["status"]=="manual_one_step_counterfactual_review_only"
    assert r["head_prediction_reproduced_for_all_signed_directions"]
    assert r["head_admission_failed"] is True
    assert r["actual_model_loss_forwards"]==39
    assert r["actual_model_context_forwards"]==1
    assert r["same_2048_lora_parameters_for_all_policies"]
    assert not r["optimizer_compute_budgets_matched"]
    assert not r["model_training_and_adapters_retained"]
    assert r["weight_promotion_permitted"] is False
    for policy in ("no_update","random_one_probe","neural_one_probe",
                   "empirical_best_eight_probes"):
        assert r["results"][policy]["retained_model_update_count"]==0


def test_actual_diagnostic_losses_do_not_support_predictive_head():
    a=assess([source(s) for s in (16,17,18)],repo_root=ROOT)
    assert a["status"]=="hold_not_qualified"
    assert a["seeds"]==[16,17,18]
    assert a["total_signed_model_candidate_perturbation_forwards"]==24
    assert a["total_actual_model_loss_forwards"]==117
    assert a["total_actual_model_context_forwards"]==3
    assert a["total_persistent_model_adapter_updates"]==0
    assert a["equal_policy_compute_budget"] is False
    assert not a["admission_to_live_optimizer_from_this_report"]
    assert not a["optimizer_switch_permitted"]
    assert not a["model_training_permitted"]
    assert not a["weight_promotion_permitted"]
    method=a["policy_comparison"]
    assert method["neural_one_probe"]["positive_train_gain_temporary_steps"]==1
    assert method["empirical_best_eight_probes"]["positive_train_gain_temporary_steps"]==2
    assert method["random_one_probe"]["positive_train_gain_temporary_steps"]==0
    assert method["neural_one_probe"]["per_seed_loss_deltas"]["18"]>0
    assert method["empirical_best_eight_probes"]["per_seed_loss_deltas"]["16"]>0
    assert method["empirical_best_eight_probes"]["per_seed_loss_deltas"]["18"]>0


@pytest.mark.parametrize("tamper",[
    lambda d:d["measured_candidate_outcomes"][0].update({"measured_train_gain":88.}),
    lambda d:d["measured_candidate_outcomes"][2].update({"predicted_train_gain_before_probe":100.}),
    lambda d:d["measured_candidate_outcomes"][1]["direction"].__setitem__(0,.2),
    lambda d:d["predicted_neural_candidate_order"].reverse(),
    lambda d:d.update({"head_novelty_guard_decision":{}}),
    lambda d:d["new_train_prompt_lfm3_context"].__setitem__(0,0.),
    lambda d:d["baseline_diagnostic"].update({"mean_completion_loss":0.}),
    lambda d:d["policy_results"]["neural_one_probe"].update({"diagnostic_loss_delta_after_minus_before":-10.}),
    lambda d:d["policy_results"]["neural_one_probe"].update({"candidate_index":7}),
    lambda d:d["policy_results"]["neural_one_probe"].update({"applied_in_memory_for_diagnostic_only":True}),
    lambda d:d["policy_results"]["empirical_best_eight_probes"].update({"counterfactual_minimum_candidate_probe_forwards":1}),
    lambda d:d["policy_results"]["no_update"].update({"diagnostic_after":None}),
    lambda d:d.update({"final_adapter_sha256":"f"*64}),
    lambda d:d.update({"persistent_adapter_updates":1}),
    lambda d:d.update({"model_backward_calls":1}),
    lambda d:d.update({"model_training_permitted":True}),
    lambda d:d.update({"weight_promotion_permitted":True}),
])
def test_false_forecast_quality_training_or_method_gain_denied(tamper):
    d=copy.deepcopy(source(18))
    tamper(d)
    r=review(d,repo_root=ROOT)
    assert r["status"]=="blocked"
    assert r["weight_promotion_permitted"] is False


def test_incomplete_seed_inventory_or_false_unique_run_denied():
    records=[source(s) for s in (16,17,18)]
    assert assess(records[:2],repo_root=ROOT)["status"]=="blocked"
    assert assess([records[0],records[0],records[2]],repo_root=ROOT)["status"]=="blocked"
    altered=copy.deepcopy(records)
    altered[1]["seed"]=16
    assert assess(altered,repo_root=ROOT)["status"]=="blocked"


def test_model_evidence_without_trusted_source_is_not_validated(tmp_path):
    assert review(source(16),repo_root=tmp_path)["status"]=="blocked"

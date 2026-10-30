"""Recompute real LFM3 in-memory one-step head-guided update receipts.

This is a structural/mathematical reviewer, NOT an independent model-run
attestation. It reconstructs the original forward-only 195-parameter head,
all 8 pre-outcome forecasts, the exact train/diagnostic examples, choice
policies and outcome statistics. It cannot authenticate that source-authored
diagnostic losses came from actual CPU execution or prove real-world gains.

Under this explicitly isolated single-step counterfactual, a neural head
which FAILED the production admission gate may be compared to random/greedy
at one bounded LoRA-B step. None of these updates are retained or promoted.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_sandbox_update_quality import (
    SCHEMA as SOURCE,NEW_SEEDS,OLD_SEEDS,SIGNED_DIRECTIONS,POLICIES,SIGMA,
    examples,digest,signed_proposals,choose_indices,
)
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_task_shift_probe import (
    rebuild_training_head,training_source_path,
)
from src.lfm3_dust_text_family_transfer_review import review as source_review
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.dust-one-step-quality-review.v1"
H64=re.compile(r"[0-9a-f]{64}\Z")
H32=re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,"model_training_permitted":False,
            "weight_promotion_permitted":False}


def finite(v,lo=-100,hi=100):
    return type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi


def close(a,b,eps=1e-8):
    return finite(a,-1e9,1e9) and finite(b,-1e9,1e9) and abs(a-b)<=eps


def review(report,*,repo_root):
    if not isinstance(report,dict):
        return blocked("missing_report")
    seed=report.get("seed")
    if type(seed) is not int or seed not in NEW_SEEDS:
        return blocked("unregistered_seed")
    old=OLD_SEEDS[NEW_SEEDS.index(seed)]
    expected={
        "schema":SOURCE,
        "status":"isolated_one_step_counterfactual_not_admitted_training",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "seed":seed,"source_head_seed":old,
        "selected_2048_lora_b_parameters":2048,
        "trained_neural_head_parameter_count":195,
        "trained_head_historical_skill_pass":False,
        "diagnostic_example_count":6,"single_train_prompt_count":1,
        "model_probe_directions_measured":SIGNED_DIRECTIONS,
        "model_candidate_perturbation_forwards":8,
        "model_clean_probe_forwards":1,
        "model_diagnostic_forwards":30,
        "total_model_loss_forwards":39,
        "model_prompt_context_forwards":1,
        "head_loss_evaluations_in_this_run":0,
        "model_backward_calls":0,"persistent_adapter_updates":0,
        "models_or_adapters_saved":0,
        "neural_head_learned_from_new_diagnostic_labels":False,
        "selection_test_data_opened":False,
        "independent_execution_attested":False,
        "counterfactuals_are_not_equal_walltime_or_energy":True,
        "this_report_is_not_a_three_way_adamw_muon_budget_match":True,
        "model_training_permitted":False,
        "optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
        "production_execution_permitted":False,
    }
    if any(report.get(k)!=v for k,v in expected.items()):
        return blocked("unapproved_model_scope_or_false_training_claim")
    if (type(report.get("run_id")) is not str
            or not H32.fullmatch(report["run_id"])
            or not finite(report.get("elapsed_s"),1e-7,1000)
            or type(report.get("initial_adapter_sha256")) is not str
            or not H64.fullmatch(report["initial_adapter_sha256"])
            or report.get("final_adapter_sha256")!=report["initial_adapter_sha256"]):
        return blocked("adapter_not_restored_or_run_identity_missing")
    try:
        path=training_source_path(Path(repo_root),old)
        if (path.is_symlink() or not path.is_file()
                or not 100<path.stat().st_size<=175000):
            return blocked("unsafe_head_training_source")
        raw=path.read_bytes()
        if report.get("source_head_training_receipt_sha256")!=hashlib.sha256(raw).hexdigest():
            return blocked("historical_head_source_digest_mismatch")
        source=json.loads(raw)
        if source_review(source).get("status")!="manual_transfer_review_only":
            return blocked("original_head_training_failed")
        head=rebuild_training_head(source)
        gate=FrozenTaskShiftGate(source=source)
        if (gate.verified_skill
                or report.get("trained_neural_head_sha256")!=head.parameter_sha256):
            return blocked("unqualified_head_or_wrong_sha")
        samples=examples(seed,source)
        manifests={name:digest(rows) for name,rows in samples.items()}
        if report.get("train_and_diagnostic_sha256")!=manifests:
            return blocked("task_or_diagnostic_manifest_changed")
        context=report.get("new_train_prompt_lfm3_context")
        if (not isinstance(context,list) or len(context)!=4
                or any(not finite(x,-1.0001,1.0001) for x in context)
                or not close(sum(x*x for x in context),1,eps=1e-5)):
            return blocked("invalid_prompt_attention_context")
        expected_guard=gate.evaluate(context,0.,candidate_index=0)
        if (report.get("head_novelty_guard_decision")!=expected_guard
                or report.get("current_train_prompt_in_envelope")!=
                   expected_guard["in_training_context_envelope"]):
            return blocked("unreproducible_training_only_head_guard")
        proposals=report.get("measured_candidate_outcomes")
        frozen_directions=signed_proposals(seed)
        if not isinstance(proposals,list) or len(proposals)!=SIGNED_DIRECTIONS:
            return blocked("missing_or_selectively_measured_directions")
        predictions=[]
        gains=[]
        for i,entry in enumerate(proposals):
            target=frozen_directions[i]
            if (not isinstance(entry,dict)
                    or entry.get("candidate")!=target["candidate"]
                    or entry.get("sign")!=target["sign"]
                    or type(entry["candidate"]) is not int
                    or type(entry["sign"]) is not int
                    or not isinstance(entry.get("direction"),list)
                    or len(entry["direction"])!=8
                    or any(not close(a,b) for a,b in zip(
                        entry["direction"],target["direction"],strict=True))):
                return blocked("candidate_not_preregistered_antithetic_direction")
            pred=head.predict(target["direction"],context,episode=12)["predicted_gain"]
            if not close(entry.get("predicted_train_gain_before_probe"),pred):
                return blocked("head_preprobe_predictions_not_reproduced")
            gain=entry.get("measured_train_gain")
            if not finite(gain):
                return blocked("unbounded_or_missing_real_model_gain")
            predictions.append(pred)
            gains.append(gain)
        expected_order=sorted(range(SIGNED_DIRECTIONS),
                              key=lambda i:(-predictions[i],i))
        if report.get("predicted_neural_candidate_order")!=expected_order:
            return blocked("head_rank_tampered_or_hindsight")
        policy_choice=choose_indices(predictions,gains,seed=seed)
        if (not isinstance(report.get("policy_results"),dict)
                or set(report["policy_results"])!=set(POLICIES)):
            return blocked("wrong_counterfactual_arm_inventory")
        before=report.get("baseline_diagnostic")
        if (not isinstance(before,dict) or
                set(before)!={"mean_completion_loss","by_family","diagnostic_model_forwards"}):
            return blocked("wrong_baseline_report_schema")
        if not _score_wellformed(before):
            return blocked("wrong_baseline_completion_loss")
        if before["diagnostic_model_forwards"]!=6:
            return blocked("wrong_baseline_compute_cost")
        total_applied=0
        results={}
        for policy in POLICIES:
            item=report["policy_results"][policy]
            if not isinstance(item,dict) or item.get("policy")!=policy:
                return blocked("wrong_method_identity")
            selected=policy_choice[policy]
            gain=gains[selected] if selected is not None else 0.
            applied=selected is not None and gain>1e-8
            if (item.get("candidate_index")!=selected
                    or item.get("applied_in_memory_for_diagnostic_only") is not applied
                    or item.get("candidate_preselected_before_train_measurement") is not
                       (policy=="neural_one_probe")
                    or not close(item.get("measured_train_gain_of_candidate"),gain)
                    or item.get("adapter_restored_sha256")!=report["initial_adapter_sha256"]
                    or item.get("adapter_exported") is not False
                    or item.get("production_update_permitted") is not False
                    or not finite(item.get("elapsed_including_diagnostics_s"),0,1000)
                    or item.get("counterfactual_minimum_candidate_probe_forwards")!=(
                        0 if selected is None else 8 if policy=="empirical_best_eight_probes" else 1)
                    or item.get("actual_diagnostic_forward_evaluations")!=6):
                return blocked("counterfactual_model_update_claim_invalid")
            after=item.get("diagnostic_after")
            if not _score_wellformed(after):
                return blocked("invalid_diagnostic_after")
            if (not close(item.get("diagnostic_loss_delta_after_minus_before"),
                          after["mean_completion_loss"]-before["mean_completion_loss"])
                    or not close(item.get("diagnostic_same_family_delta"),
                          after["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                          before["by_family"]["diagnostic_same_family"]["mean_completion_loss"])
                    or not close(item.get("diagnostic_transfer_delta"),
                          after["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                          before["by_family"]["diagnostic_transfer"]["mean_completion_loss"])):
                return blocked("incorrect_quality_gain_or_hindsight")
            if policy=="no_update" and any(
                    not close(after["by_family"][name]["mean_completion_loss"],
                              before["by_family"][name]["mean_completion_loss"])
                    for name in ("diagnostic_same_family","diagnostic_transfer")):
                return blocked("no_update_changed_quality")
            total_applied+=int(applied)
            results[policy]={
                "candidate_index":selected,"positive_train_gain":gain>1e-8,
                "isolated_adapter_step_evaluated":applied,
                "diagnostic_loss_delta":item["diagnostic_loss_delta_after_minus_before"],
                "same_family_loss_delta":item["diagnostic_same_family_delta"],
                "transfer_family_loss_delta":item["diagnostic_transfer_delta"],
                "counterfactual_probe_budget":item["counterfactual_minimum_candidate_probe_forwards"],
                "retained_model_update_count":0,
            }
        return {
            "schema":SCHEMA,"status":"manual_one_step_counterfactual_review_only",
            "seed":seed,"train_and_diagnostic_manifest":manifests,
            "head_attended_from_prior_feedback":True,
            "head_admission_failed":True,
            "head_prediction_reproduced_for_all_signed_directions":True,
            "head_preprobe_top1_direction_index":policy_choice["neural_one_probe"],
            "isolated_policy_updates_temporarily_evaluated":total_applied,
            "actual_model_loss_forwards":report["total_model_loss_forwards"],
            "actual_model_context_forwards":1,
            "same_2048_lora_parameters_for_all_policies":True,
            "optimizer_compute_budgets_matched":False,
            "model_training_and_adapters_retained":False,
            "source_runtime_independently_attested":False,
            "results":results,
            "optimizer_switch_permitted":False,
            "model_training_permitted":False,
            "weight_promotion_permitted":False,
        }
    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,
            ZeroDivisionError,RuntimeError,json.JSONDecodeError):
        return blocked("unreconstructible_one_step_training_or_diagnostics")


def _score_wellformed(result):
    if not isinstance(result,dict) or set(result)!={
            "mean_completion_loss","by_family","diagnostic_model_forwards"}:
        return False
    family=result["by_family"]
    if not isinstance(family,dict) or set(family)!={
            "diagnostic_same_family","diagnostic_transfer"}:
        return False
    for name,count in (("diagnostic_same_family",4),("diagnostic_transfer",2)):
        r=family[name]
        if not isinstance(r,dict) or set(r)!={"count","mean_completion_loss"}:
            return False
        if type(r["count"]) is not int or r["count"]!=count:
            return False
        if not finite(r["mean_completion_loss"],0,100):
            return False
    mean=(4*family["diagnostic_same_family"]["mean_completion_loss"]+
          2*family["diagnostic_transfer"]["mean_completion_loss"])/6
    return close(mean,result["mean_completion_loss"]) and (
        type(result["diagnostic_model_forwards"]) is int and
        result["diagnostic_model_forwards"]==6)

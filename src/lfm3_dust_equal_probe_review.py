"""Deterministic replay of actual equal-probe LFM3 head-v-random trials.

Reconstruct the frozen trained external attention network from historical
training receipts, all 8 pre-probe forecasts, fixed candidate identities and
one-model-probe budget for each policy. Validate diagnostic loss deltas,
adapter-hash restoration and the no-training/no-promotion contract.

Source-run model losses and walltime are locally authored JSON, NOT signed
hardware execution telemetry; this checks math/custody consistency only.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_equal_probe_benchmark import (
    SCHEMA as SOURCE, SEEDS, HEAD_SEEDS, POLICIES,
    task_examples,candidate_set,choose,digest,
)
from src.lfm3_dust_task_shift_probe import rebuild_training_head,training_source_path
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_text_family_transfer_review import review as verify_training
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.equal-probe-review.v1"
HASH=re.compile(r"[a-f0-9]{64}\Z")
IDENTITY=re.compile(r"[a-f0-9]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,"model_update_permitted":False,
            "weight_promotion_permitted":False}


def finite(value,minimum=-100,maximum=100):
    return type(value) in (int,float) and math.isfinite(value) and minimum<=value<=maximum


def close(a,b,tol=1e-8):
    return finite(a,-1e9,1e9) and finite(b,-1e9,1e9) and abs(a-b)<=tol


def _diag(d):
    if (not isinstance(d,dict) or set(d)!={
            "mean_completion_loss","by_family","diagnostic_model_forwards"}
            or type(d["diagnostic_model_forwards"]) is not int
            or d["diagnostic_model_forwards"]!=6):
        return False
    x=d["by_family"]
    if not isinstance(x,dict) or set(x)!={
            "diagnostic_same_family","diagnostic_transfer"}:
        return False
    for name,n in (("diagnostic_same_family",4),("diagnostic_transfer",2)):
        row=x[name]
        if (not isinstance(row,dict) or set(row)!={"count","mean_completion_loss"}
                or type(row["count"]) is not int or row["count"]!=n
                or not finite(row["mean_completion_loss"],0,100)):
            return False
    mean=(4*x["diagnostic_same_family"]["mean_completion_loss"]+
          2*x["diagnostic_transfer"]["mean_completion_loss"])/6
    return close(mean,d["mean_completion_loss"])


def review(report,*,repo_root):
    if not isinstance(report,dict):
        return blocked("not_a_model_source")
    seed=report.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        return blocked("unregistered_equal_probe_seed")
    old=HEAD_SEEDS[SEEDS.index(seed)]
    fixed={
        "schema":SOURCE,"status":"read_only_equal_one_probe_model_counterfactual",
        "seed":seed,"source_head_seed":old,"model":"LiquidAI/LFM3-1.7B",
        "base_sha256":BASE,"head_parameters":195,
        "head_historical_gain_skill_qualified":False,
        "model_lora_b_scope":2048,
        "training_prompt_count":1,"diagnostic_prompt_count":6,
        "diagnostic_partitions":["diagnostic_same_family","diagnostic_transfer"],
        "measured_training_candidate_forwards":2,
        "measured_training_baseline_forwards":1,
        "measured_diagnostic_model_forwards":18,
        "total_loss_model_forwards":21,
        "neural_extra_prompt_context_forwards":1,
        "random_extra_prompt_context_forwards":0,
        "actual_budget_per_policy_model_probe_forwards":1,
        "actual_budget_per_policy_diagnostic_forwards":6,
        "policy_model_probe_and_diagnostic_budgets_equal":True,
        "model_load_and_shared_baseline_cost_not_per_policy":True,
        "policy_execution_order":list(POLICIES),
        "observed_walltime_not_energy_normalized":True,
        "neural_head_receipt_custody_independently_attested":False,
        "new_model_trial_custody_independently_attested":False,
        "original_locked_selection_test_opened":False,
        "head_fine_tuned_on_new_trial_outcomes":False,
        "model_backward_count":0,"persistent_lora_update_count":0,
        "model_checkpoint_saved":False,"model_training_permitted":False,
        "optimizer_switch_permitted":False,"weight_promotion_permitted":False,
    }
    if any(report.get(k)!=value for k,value in fixed.items()):
        return blocked("wrong_compute_budget_or_model_training_scope")
    if (type(report.get("run_id")) is not str
            or not IDENTITY.fullmatch(report["run_id"])
            or type(report.get("initial_adapter_sha256")) is not str
            or not HASH.fullmatch(report["initial_adapter_sha256"])
            or report.get("final_adapter_sha256")!=report["initial_adapter_sha256"]):
        return blocked("unrestored_lora_adapter_or_runtime_identity")
    try:
        path=training_source_path(Path(repo_root),old)
        if (path.is_symlink() or not path.is_file()
                or not 100<path.stat().st_size<=175000):
            return blocked("source_training_unavailable")
        data=path.read_bytes()
        if report.get("original_head_receipt_sha256")!=hashlib.sha256(data).hexdigest():
            return blocked("original_train_evidence_mismatch")
        src=json.loads(data)
        if verify_training(src).get("status")!="manual_transfer_review_only":
            return blocked("source_head_not_verified")
        head=rebuild_training_head(src)
        gate=FrozenTaskShiftGate(source=src)
        if (gate.verified_skill or
                report.get("head_frozen_sha256")!=head.parameter_sha256
                or report.get("head_training_memory_size")!=len(head.history)):
            return blocked("head_weights_or_training_feedback_changed")
        task=task_examples(seed)
        if report.get("task_sha256_manifests")!={
                name:digest(rows) for name,rows in task.items()}:
            return blocked("mismatched_training_or_independent_diagnostics")
        context=report.get("preprobe_context")
        if (not isinstance(context,list) or len(context)!=4
                or any(not finite(v,-1.001,1.001) for v in context)
                or not close(sum(v*v for v in context),1.,tol=1e-5)
                or report.get("preprobe_context_gate")!=
                   gate.evaluate(context,0.,candidate_index=0)):
            return blocked("unverified_source_lfm3_attention_context")
        proposals=candidate_set(seed)
        reported_directions=report.get("candidate_directions")
        if (not isinstance(reported_directions,list)
                or len(reported_directions)!=8):
            return blocked("missing_or_selectively_changed_candidates")
        for p,original in zip(reported_directions,proposals,strict=True):
            if (not isinstance(p,dict) or set(p)!=set(original)
                    or p.get("pair")!=original["pair"]
                    or p.get("sign")!=original["sign"]
                    or not isinstance(p.get("direction"),list)
                    or len(p["direction"])!=8
                    or any(not close(a,b) for a,b in zip(
                        p["direction"],original["direction"],strict=True))):
                return blocked("wrong_signed_training_candidate")
        forecasts=[head.predict(p["direction"],context,episode=12)["predicted_gain"]
                   for p in proposals]
        rec_pred=report.get("all_preprobe_neural_gain_forecasts")
        if (not isinstance(rec_pred,list) or len(rec_pred)!=8
                or any(not close(a,b) for a,b in zip(rec_pred,forecasts,strict=True))):
            return blocked("unreplayable_original_neural_forecasts")
        chosen=choose(seed,forecasts)
        if report.get("selected_indices_from_prehistory")!=chosen:
            return blocked("model_gain_leak_into_direction_selection")
        baseline=report.get("baseline_diagnostic")
        if not _diag(baseline):
            return blocked("wrong_baseline_or_diagnostic_samples")
        clean=report.get("clean_training_loss")
        if not finite(clean,0,100):
            return blocked("nonfinite_clean_training_completion_loss")
        arms=report.get("policy_results")
        if not isinstance(arms,dict) or set(arms)!=set(POLICIES):
            return blocked("missing_matched_one_probe_control")
        ordered={}
        for policy in POLICIES:
            row=arms[policy]
            index=chosen[policy]
            if not isinstance(row,dict) or row.get("policy")!=policy:
                return blocked("wrong_policy_identity")
            if (row.get("selected_signed_candidate")!=index
                    or type(row["selected_signed_candidate"]) is not int
                    or row.get("selected_direction")!=proposals[index]
                    or not close(row.get("selected_prediction_before_any_probe"),
                                 forecasts[index])
                    or not close(row.get("training_clean_completion_loss"),clean)
                    or not finite(row.get("training_perturbed_completion_loss"),0,100)):
                return blocked("head_choice_or_single_probe_not_reproducible")
            train_gain=clean-row["training_perturbed_completion_loss"]
            accepted=train_gain>1e-8
            score=row.get("diagnostic_after")
            if (not close(row.get("measured_training_gain"),train_gain)
                    or row.get("positive_gain_acceptance") is not accepted
                    or row.get("temporary_adapter_step_diagnosed") is not accepted
                    or not _diag(score)):
                return blocked("incorrect_admission_or_diagnostic_score")
            if not accepted and (not close(
                    score["mean_completion_loss"],baseline["mean_completion_loss"])
                    or any(not close(
                       score["by_family"][name]["mean_completion_loss"],
                       baseline["by_family"][name]["mean_completion_loss"])
                       for name in ("diagnostic_same_family","diagnostic_transfer"))):
                return blocked("abstention_cannot_change_model_quality")
            if (not close(
                    row.get("diagnostic_completion_loss_delta"),
                    score["mean_completion_loss"]-baseline["mean_completion_loss"])
                    or not close(row.get("diagnostic_same_family_loss_delta"),
                        score["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                        baseline["by_family"]["diagnostic_same_family"]["mean_completion_loss"])
                    or not close(row.get("diagnostic_transfer_loss_delta"),
                        score["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                        baseline["by_family"]["diagnostic_transfer"]["mean_completion_loss"])
                    or row.get("model_probe_forwards")!=1
                    or row.get("model_diagnostic_forwards")!=6
                    or row.get("model_backwards")!=0
                    or row.get("adapter_restored_sha256")!=report["initial_adapter_sha256"]
                    or row.get("retained_model_update") is not False
                    or row.get("checkpoint_saved") is not False
                    or not finite(row.get("policy_probe_and_diagnostic_wall_s"),1e-9,1000)):
                return blocked("fake_equal_compute_or_diagnostic_quality")
            ordered[policy]={
                "candidate_index":index,"training_gain":train_gain,
                "temporary_adapter_step_accepted":accepted,
                "diagnostic_loss_delta":row["diagnostic_completion_loss_delta"],
                "same_family_loss_delta":row["diagnostic_same_family_loss_delta"],
                "transfer_loss_delta":row["diagnostic_transfer_loss_delta"],
                "probe_plus_diagnostic_elapsed_s":row["policy_probe_and_diagnostic_wall_s"],
            }
        for key in ("neural_head_reconstruction_elapsed_s",
                    "neural_prompt_context_elapsed_s",
                    "neural_head_forecast_elapsed_s",
                    "total_experiment_inference_elapsed_s"):
            if not finite(report.get(key),1e-9,3600):
                return blocked("missing_compute_timing_provenance")
        return {
            "schema":SCHEMA,"status":"manual_matched_probe_budget_review_only",
            "seed":seed,"selected_lora_parameters":2048,
            "model_probe_forwards_each_policy":1,
            "diagnostic_forwards_each_policy":6,
            "shared_baseline_loss_forwards":7,
            "model_prompt_context_forwards_neural":1,
            "neural_head_reconstruction_timing_separately_reported":True,
            "actual_training_model_loss_forwards":21,
            "head_admission_failed":True,
            "model_weights_and_external_head_unchanged":True,
            "results":ordered,
            "source_runtime_independently_attested":False,
            "optimizer_switch_permitted":False,
            "model_update_permitted":False,
            "weight_promotion_permitted":False,
        }
    except (OSError,ValueError,TypeError,KeyError,IndexError,
            RuntimeError,OverflowError,json.JSONDecodeError,ZeroDivisionError):
        return blocked("unreconstructible_equal_budget_probe_evidence")

"""Strict causal replay of real pretrained LFM3 one-feedback adaptive Q/K head.

Rebuild all prior 195-weight forward-only training, then replay each
subsequent head-only weight proposal using only the previous episode's
ONE measured neural-selected training gain. Recompute all 8 preprobe
forecasts, selections, model diagnostic arithmetic and adapter restoration.
This does not independently cryptographically attest the LFM3 source run.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_online_one_feedback_trial import (
    SCHEMA as SOURCE,SEEDS,HEAD_SEEDS,EPISODES,POLICIES,tasks,candidates,
    policy_choice,digest,
)
from src.lfm3_dust_online_single_feedback import OnlineSingleFeedbackHead
from src.lfm3_dust_task_shift_probe import training_source_path,rebuild_training_head
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_text_family_transfer_review import review as original_review
from src.lfm3_dust_equal_probe_review import _diag
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.one-feedback-online-head-review.v1"
SHA=re.compile(r"[0-9a-f]{64}\Z")
ID=re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "model_training_permitted":False,"optimizer_switch_permitted":False,
            "weight_promotion_permitted":False}


def near(a,b,eps=1e-8):
    return (type(a) in (int,float) and type(b) in (int,float)
            and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=eps)


def review(report,*,repo_root):
    if not isinstance(report,dict):
        return blocked("not_a_runtime_report")
    seed=report.get("seed")
    if type(seed) is not int or seed not in SEEDS:
        return blocked("unregistered_one_feedback_seed")
    old=HEAD_SEEDS[SEEDS.index(seed)]
    exact={
        "schema":SOURCE,"status":"offline_online_one_label_head_research_only",
        "seed":seed,"source_head_seed":old,
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "online_head_parameters":195,"online_training_gain_observations":3,
        "trained_head_uses_diagnostic_feedback":False,
        "model_parameters_eligible_for_temporary_step":2048,
        "total_episodes":3,"policy_names":list(POLICIES),
        "candidate_model_probes_per_policy_per_episode":1,
        "diagnostic_model_forwards_per_policy_per_episode":6,
        "total_model_completion_loss_forwards":84,
        "total_model_prompt_context_forwards":3,
        "training_model_backward_calls":0,
        "persistent_model_adapter_updates":0,
        "model_checkpoints_saved":0,
        "selection_test_partition_opened":False,
        "head_proposals_are_backprop_free":True,
        "model_source_custody_independently_attested":False,
        "compute_budget_equal_in_training_probes_but_not_total":True,
        "model_update_permitted":False,
        "optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
    }
    if any(report.get(k)!=v for k,v in exact.items()):
        return blocked("false_model_budget_or_promotion_evidence")
    if (type(report.get("run_id")) is not str
            or not ID.fullmatch(report["run_id"])
            or type(report.get("model_adapter_initial_sha256")) is not str
            or not SHA.fullmatch(report["model_adapter_initial_sha256"])
            or report.get("model_adapter_final_sha256")!=
               report["model_adapter_initial_sha256"]
            or not near(report.get("model_inference_total_elapsed_s"),
                        report.get("model_inference_total_elapsed_s"))
            or report["model_inference_total_elapsed_s"]<=0):
        return blocked("model_not_restored_or_invalid_run_receipt")
    try:
        original_path=training_source_path(Path(repo_root),old)
        if (original_path.is_symlink() or not original_path.is_file()
                or not 100<original_path.stat().st_size<=175000):
            return blocked("unsafe_original_head_source")
        raw=original_path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=report.get("source_head_receipt_sha256"):
            return blocked("historical_head_source_changed")
        source=json.loads(raw)
        if original_review(source).get("status")!="manual_transfer_review_only":
            return blocked("original_head_training_not_replayable")
        fixed=rebuild_training_head(source)
        online=OnlineSingleFeedbackHead(rebuild_training_head(source),seed=seed)
        if (FrozenTaskShiftGate(source=source).verified_skill
                or report.get("frozen_start_head_sha256")!=fixed.parameter_sha256
                or report.get("online_start_head_sha256")!=online.head.parameter_sha256):
            return blocked("unqualified_head_training_history_changed")
        items=tasks(seed)
        manifests=[{"episode":row["episode"],
                    "training":digest(row["training"]),
                    "diagnostic_same_family":digest(row["diagnostic_same_family"]),
                    "diagnostic_transfer":digest(row["diagnostic_transfer"])}
                   for row in items]
        if report.get("sample_manifests")!=manifests:
            return blocked("changed_training_or_diagnostic_prompt_manifest")
        rows=report.get("episodes")
        if not isinstance(rows,list) or len(rows)!=len(EPISODES):
            return blocked("missing_or_selectively_reported_episodes")
        result=[]
        total_accepted=0
        for ep,data in zip(EPISODES,rows,strict=True):
            if (not isinstance(data,dict) or type(data.get("episode")) is not int
                    or data["episode"]!=ep
                    or data.get("task_manifests")!=manifests[ep-12]
                    or data.get("model_loss_forwards")!=28
                    or data.get("prompt_context_model_forwards")!=1
                    or data.get("model_backward_count")!=0
                    or data.get("persistent_model_updates")!=0
                    or data.get("frozen_head_sha256")!=fixed.parameter_sha256):
                return blocked("untrusted_episode_model_or_head_state")
            expected_train=online.prepare(episode=ep,adaptive=True)
            if data.get("training_only_head_update_before_current_forecasts")!=expected_train:
                return blocked("head_lookahead_or_wrong_derivative_free_update")
            if data.get("adaptive_head_sha_after_training_only")!=online.head.parameter_sha256:
                return blocked("adaptive_qk_hash_mismatch")
            context=data.get("preprobe_context")
            if (not isinstance(context,list) or len(context)!=4
                    or any(type(x) not in (int,float) or not math.isfinite(x)
                           for x in context)
                    or not near(sum(x*x for x in context),1,eps=1e-5)):
                return blocked("invalid_frozen_lfm3_attention_context")
            dirs=candidates(seed,ep)
            recorded=data.get("signed_directions")
            if (not isinstance(recorded,list) or len(recorded)!=8
                    or any(not isinstance(x,dict) or x.get("pair")!=y["pair"]
                           or x.get("sign")!=y["sign"]
                           or not isinstance(x.get("direction"),list)
                           or len(x["direction"])!=8
                           or any(not near(a,b) for a,b in zip(x["direction"],y["direction"],strict=True))
                           for x,y in zip(recorded,dirs,strict=True))):
                return blocked("preselected_signed_directions_tampered")
            predicted_fixed=[fixed.predict(x["direction"],context,episode=ep)[
                "predicted_gain"] for x in dirs]
            predicted_online=online.forecast([x["direction"] for x in dirs],
                                              context,episode=ep)
            for key,expected_preds in (("frozen_forecasts_before_any_probe",predicted_fixed),
                                       ("adaptive_forecasts_before_any_probe",predicted_online)):
                got=data.get(key)
                if (not isinstance(got,list) or len(got)!=8
                        or any(not near(x,y) for x,y in zip(got,expected_preds,strict=True))):
                    return blocked("preprobe_neural_head_forecasts_falsified")
            choice=policy_choice(seed,ep,predicted_fixed,predicted_online)
            if data.get("candidate_policy_choice")!=choice:
                return blocked("current_outcome_used_to_select_head_update")
            baseline=data.get("baseline_diagnostic")
            clean=data.get("clean_training_loss")
            if not _diag(baseline) or not near(clean,clean) or not 0<=clean<=100:
                return blocked("invalid_baseline_diagnostic_or_training_loss")
            arms=data.get("policy_results")
            if not isinstance(arms,dict) or set(arms)!=set(POLICIES):
                return blocked("incomplete_equal_model_probe_policy_inventory")
            episode_summary={}
            for policy in POLICIES:
                measured=arms[policy]
                index=choice[policy]
                if (not isinstance(measured,dict)
                        or measured.get("selected_index")!=index
                        or type(measured["selected_index"]) is not int
                        or not near(measured.get("train_clean_loss"),clean)
                        or not near(measured.get("train_perturbed_loss"),
                                    measured.get("train_perturbed_loss"))):
                    return blocked("wrong_probed_policy_or_training_measurement")
                gain=clean-measured["train_perturbed_loss"]
                accepted=gain>1e-8
                score=measured.get("diagnostic_after")
                if (not near(gain,measured.get("train_gain"))
                        or measured.get("training_gain_positive") is not accepted
                        or not _diag(score)):
                    return blocked("unreproducible_train_only_gain_or_diagnostic")
                if policy=="adaptive_head_one_probe":
                    observe=online.observe(dirs[index]["direction"],
                        context,gain,episode=ep)
                    if measured.get("one_train_only_observation")!=observe:
                        return blocked("single_train_feedback_not_causal")
                elif measured.get("one_train_only_observation") is not None:
                    return blocked("other_policy_used_to_train_the_head")
                if not accepted and (
                        not near(score["mean_completion_loss"],
                                 baseline["mean_completion_loss"]) or
                        any(not near(score["by_family"][key]["mean_completion_loss"],
                                     baseline["by_family"][key]["mean_completion_loss"])
                            for key in ("diagnostic_same_family","diagnostic_transfer"))):
                    return blocked("abstained_model_has_modified_diagnostics")
                if (not near(measured.get("mean_diagnostic_loss_delta"),
                             score["mean_completion_loss"]-baseline["mean_completion_loss"])
                        or not near(measured.get("same_family_loss_delta"),
                             score["by_family"]["diagnostic_same_family"]["mean_completion_loss"]-
                             baseline["by_family"]["diagnostic_same_family"]["mean_completion_loss"])
                        or not near(measured.get("transfer_loss_delta"),
                             score["by_family"]["diagnostic_transfer"]["mean_completion_loss"]-
                             baseline["by_family"]["diagnostic_transfer"]["mean_completion_loss"])
                        or measured.get("model_probe_forwards")!=1
                        or measured.get("model_diagnostic_forwards")!=6
                        or measured.get("model_backwards")!=0
                        or measured.get("adapter_sha_after_restoring")!=
                           report["model_adapter_initial_sha256"]
                        or measured.get("checkpoint_saved") is not False
                        or measured.get("production_execution_permitted") is not False
                        or type(measured.get("policy_elapsed_seconds")) not in (int,float)
                        or not 0<measured["policy_elapsed_seconds"]<300):
                    return blocked("wrong_post_update_diagnostic_or_model_budget")
                episode_summary[policy]={
                    "selected_index":index,
                    "training_gain":gain,
                    "temporary_adapter_step":accepted,
                    "diagnostic_loss_delta":measured["mean_diagnostic_loss_delta"],
                    "transfer_loss_delta":measured["transfer_loss_delta"],
                }
            for key in ("head_training_seconds","context_model_forward_seconds",
                        "both_heads_forecast_seconds","episode_inference_elapsed_s"):
                if (type(data.get(key)) not in (int,float)
                        or not 0<data[key]<1200):
                    return blocked("invalid_head_or_inference_cost_measurement")
            total_accepted+=expected_train["head_only_accepted_updates"]
            result.append({"episode":ep,"head_updates_before_forecast":
                           expected_train["head_only_accepted_updates"],
                           "head_only_objectives":expected_train["head_only_objective_forwards"],
                           "qk_head_sha256":online.head.parameter_sha256,
                           "policies":episode_summary})
        if (report.get("frozen_end_head_sha256")!=fixed.parameter_sha256
                or report.get("online_end_head_sha256")!=online.head.parameter_sha256
                or report.get("online_head_only_accepted_updates")!=total_accepted
                or report.get("online_head_only_objective_forwards")!=
                   online.total_head_only_objective_forwards
                or not near(report.get("online_head_l2_drift"),online.head.update_norm)
                or online.head.update_norm>.6+1e-9):
            return blocked("online_training_count_or_final_head_hash_changed")
        return {
            "schema":SCHEMA,"status":"manual_causal_one_feedback_head_review_only",
            "seed":seed,
            "head_qk_weights_learn_from_prior_single_train_gains":True,
            "real_pretrained_model_loss_forwards":84,
            "real_model_prompt_context_forwards":3,
            "total_online_head_only_objectives":online.total_head_only_objective_forwards,
            "total_online_head_weight_updates":total_accepted,
            "final_online_qk_sha256":online.head.parameter_sha256,
            "frozen_head_unchanged":True,
            "same_2048_model_lora_scope":True,
            "model_persistent_adapter_updates":0,
            "model_runtime_independently_attested":False,
            "episodes":result,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
        }
    except (OSError,ValueError,TypeError,KeyError,IndexError,RuntimeError,
            OverflowError,json.JSONDecodeError,ZeroDivisionError):
        return blocked("bad_or_unreproducible_single_feedback_model_run")

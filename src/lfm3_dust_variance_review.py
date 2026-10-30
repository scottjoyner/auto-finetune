"""Deterministic fail-closed review of 3-prompt LFM3 Dust counterfactual.

Reconstruct historical 195-weight forward-only trained head, every prior
candidate prediction, registered directions, deterministic random/greedy
selection, measured positive-gain confirmations, and per-policy diagnostic
deltas. Checks no persistent LoRA updates. This is MATHEMATICAL validation
of source-authored telemetry, not independent attestation of LFM3 execution.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_variance_quality import (
    SCHEMA as SOURCE, NEW_SEEDS, HEAD_SEEDS, CANDIDATES, POLICIES,
    samples_for, proposals_for, choices, verdicts, variance_verdict, digest,
)
from src.lfm3_dust_task_shift_probe import (
    rebuild_training_head, training_source_path,
)
from src.lfm3_dust_task_shift_gate import FrozenTaskShiftGate
from src.lfm3_dust_text_family_transfer_review import review as original_review
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.dust-variance-quality-review.v1"
SHA=re.compile(r"[0-9a-f]{64}\Z")
RUN=re.compile(r"[0-9a-f]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "optimizer_switch_permitted":False,
            "model_update_permitted":False,"weight_promotion_permitted":False}


def near(x,y,eps=1e-8):
    return type(x) in (int,float) and type(y) in (int,float) and (
        math.isfinite(x) and math.isfinite(y) and abs(x-y)<=eps)


def diagnostic_ok(d):
    if (not isinstance(d,dict) or set(d)!={
            "mean_completion_loss","by_family","diagnostic_model_forwards"}
            or d["diagnostic_model_forwards"]!=6):
        return False
    families=d["by_family"]
    if not isinstance(families,dict) or set(families)!={
            "diagnostic_same_family","diagnostic_transfer"}:
        return False
    for key,count in (("diagnostic_same_family",4),("diagnostic_transfer",2)):
        row=families[key]
        if (not isinstance(row,dict) or set(row)!={"count","mean_completion_loss"}
                or type(row["count"]) is not int or row["count"]!=count
                or not near(row["mean_completion_loss"],
                            row["mean_completion_loss"])
                or not 0<=row["mean_completion_loss"]<=100):
            return False
    v=(4*families["diagnostic_same_family"]["mean_completion_loss"]+
       2*families["diagnostic_transfer"]["mean_completion_loss"])/6
    return near(v,d["mean_completion_loss"])


def review(d,*,repo_root):
    if not isinstance(d,dict):
        return blocked("not_a_source_report")
    seed=d.get("seed")
    if type(seed) is not int or seed not in NEW_SEEDS:
        return blocked("unregistered_multi_prompt_seed")
    head_seed=HEAD_SEEDS[NEW_SEEDS.index(seed)]
    expected={
        "schema":SOURCE,
        "status":"offline_unqualified_head_variance_counterfactual",
        "seed":seed,"trained_head_seed":head_seed,
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "trained_attention_parameters":195,
        "variance_gate_definition":"all_three_positive_and_sample_mean_minus_one_sample_std_positive",
        "variance_acceptance_is_unverified_heuristic":True,
        "selected_model_adapter_parameters":2048,
        "training_prompts":3,"diagnostic_prompts":6,
        "historical_head_accepted_by_gain_skill":False,
        "model_gradients_and_backprop_calls":0,
        "head_training_updates_this_run":0,
        "persistent_model_adapter_updates":0,
        "models_or_adapters_saved":0,
        "selection_test_partition_opened":False,
        "source_runtime_independently_attested":False,
        "diagnostic_labels_used_for_head_or_gate":False,
        "candidate_direction_count":8,
        "candidate_first_prompt_forwards":8,
        "training_clean_model_forwards":3,
        "diagnostic_baseline_model_forwards":6,
        "total_model_prompt_context_forwards":1,
        "comparison_is_not_compute_matched_to_adamw_muon":True,
        "single_and_multi_comparison_reuses_identical_candidate_outcomes":True,
        "production_optimizer_switch_permitted":False,
        "model_update_permitted":False,
        "weight_promotion_permitted":False,
    }
    if any(d.get(k)!=v for k,v in expected.items()):
        return blocked("unauthorized_training_or_runtime_claim")
    if (type(d.get("run_id")) is not str or not RUN.fullmatch(d["run_id"])
            or type(d.get("initial_adapter_sha256")) is not str
            or not SHA.fullmatch(d["initial_adapter_sha256"])
            or d.get("final_adapter_sha256")!=d["initial_adapter_sha256"]
            or type(d.get("elapsed_s")) not in (int,float)
            or not 0<d["elapsed_s"]<1500):
        return blocked("unverified_adapter_restore_or_run")
    try:
        path=training_source_path(Path(repo_root),head_seed)
        if (path.is_symlink() or not path.is_file()
                or not 100<path.stat().st_size<=175000):
            return blocked("untrusted_original_head_source")
        original=path.read_bytes()
        if hashlib.sha256(original).hexdigest()!=d.get("source_training_receipt_sha256"):
            return blocked("head_source_sha_mismatch")
        historical=json.loads(original)
        if original_review(historical).get("status")!="manual_transfer_review_only":
            return blocked("original_neural_training_not_validated")
        head=rebuild_training_head(historical)
        gate=FrozenTaskShiftGate(source=historical)
        if (gate.verified_skill or
                head.parameter_sha256!=d.get("source_training_head_sha256")
                or len(head.history)!=d.get("source_head_memory_entries")):
            return blocked("head_or_historical_admission_changed")
        examples=samples_for(seed)
        if d.get("train_diagnostic_manifest")!={
                name:digest(items) for name,items in examples.items()}:
            return blocked("different_data_partition_or_diagnostic_samples")
        ctx=d.get("new_train_prompt_attention_context")
        if (not isinstance(ctx,list) or len(ctx)!=4
                or any(not near(x,x) or not -1.0001<=x<=1.0001
                       for x in ctx)
                or not near(sum(x*x for x in ctx),1.,eps=1e-5)
                or d.get("train_only_shift_guard_decision")!=
                   gate.evaluate(ctx,0.,candidate_index=0)):
            return blocked("invalid_original_preprobe_lfm3_activations")
        entries=d.get("proposals")
        expected_dirs=proposals_for(seed)
        if not isinstance(entries,list) or len(entries)!=CANDIDATES:
            return blocked("incomplete_signed_candidate_probes")
        forecasts=[]
        gains=[]
        for entry,expected_dir in zip(entries,expected_dirs,strict=True):
            if (not isinstance(entry,dict)
                    or set(entry)!=set(expected_dir)|{
                        "predicted_gain_before_any_measurement",
                        "measured_first_train_gain"}
                    or entry.get("candidate")!=expected_dir["candidate"]
                    or entry.get("sign")!=expected_dir["sign"]
                    or not isinstance(entry.get("direction"),list)
                    or len(entry["direction"])!=8
                    or any(not near(a,b) for a,b in zip(
                        entry["direction"],expected_dir["direction"],strict=True))):
                return blocked("bad_antithetic_direction_or_provenance")
            forecast=head.predict(expected_dir["direction"],ctx,episode=12)["predicted_gain"]
            if not near(forecast,entry.get("predicted_gain_before_any_measurement")):
                return blocked("neural_forecast_contains_hindsight_or_wrong_head")
            gain=entry.get("measured_first_train_gain")
            if not near(gain,gain):
                return blocked("nonfinite_first_prompt_training_gain")
            forecasts.append(forecast)
            gains.append(gain)
        selected=choices(forecasts,gains,seed=seed)
        if d.get("selected_candidate_index_by_policy")!=selected:
            return blocked("selected_direction_not_preregistered")
        all_positive={idx for idx in selected.values() if gains[idx]>1e-8}
        confirms=d.get("candidate_confirmation_gains")
        quality=d.get("positive_first_gain_candidate_diagnostics")
        if (not isinstance(confirms,dict) or
                set(confirms)!={str(idx) for idx in set(selected.values())}
                or not isinstance(quality,dict) or
                set(quality)!={str(i) for i in all_positive}):
            return blocked("incomplete_or_hindsight_confirmations")
        for idx in set(selected.values()):
            result=confirms[str(idx)]
            if idx not in all_positive and result is not None:
                return blocked("confirmation_probes_should_have_been_skipped")
            if idx in all_positive and (not isinstance(result,list)
                    or len(result)!=2
                    or any(not near(x,x) or abs(x)>100 for x in result)):
                return blocked("missing_two_independent_training_confirmations")
        baseline=d.get("baseline_diagnostic")
        if not diagnostic_ok(baseline):
            return blocked("invalid_baseline_diagnostic")
        for item in quality.values():
            if not diagnostic_ok(item):
                return blocked("invalid_candidate_diagnostics")
        if not isinstance(d.get("results"),dict) or set(d["results"])!=set(POLICIES):
            return blocked("missing_single_multi_arm")
        if d["results"]["no_update"]!={
                "candidate_index":None,"training_gain_first":0.,
                "training_confirmation_gains":None,
                "single_prompt_would_apply":False,"multi_prompt_would_apply":False,
                "diagnostic_after":baseline,"diagnostic_mean_loss_delta":0.,
                "guard_reason":"no_update","retained_adapter_updates":0}:
            return blocked("no_update_baseline_modified")
        n_veto,n_harm=0,0
        rows={}
        for policy in ("random","neural","greedy"):
            idx=selected[policy]
            decision=verdicts(gains[idx],confirms[str(idx)])
            decision["variance"]=variance_verdict(gains[idx],confirms[str(idx)])
            if decision["single"] and not decision["multi"]:
                n_veto+=1
            if decision["single"] and (
                    quality[str(idx)]["mean_completion_loss"]-
                    baseline["mean_completion_loss"])>0:
                n_harm+=1
            for mode in ("single","multi","variance"):
                name=f"{policy}_{mode}"
                original_result=d["results"][name]
                if not isinstance(original_result,dict):
                    return blocked("invalid_policy_row")
                applied=decision[mode]
                after=quality[str(idx)] if applied else baseline
                expected_result={
                    "candidate_index":idx,
                    "training_gain_first":gains[idx],
                    "training_confirmation_gains":confirms[str(idx)],
                    "single_prompt_would_apply":decision["single"],
                    "multi_prompt_would_apply":decision["multi"],
                    "would_apply_under_own_rule":applied,
                    "diagnostic_after":after,
                    "diagnostic_mean_loss_delta":after["mean_completion_loss"]-
                                                 baseline["mean_completion_loss"],
                    "guard_reason":decision["reason"],
                    "retained_adapter_updates":0,
                }
                if set(original_result)!=set(expected_result) or any(
                        not near(original_result[k],value) if type(value) is float
                        else original_result[k]!=value
                        for k,value in expected_result.items()):
                    return blocked("policy_decision_or_diagnostic_recomputed_differently")
                rows[name]={
                    "candidate":idx,
                    "applied":applied,
                    "diagnostic_loss_change":original_result["diagnostic_mean_loss_delta"],
                    "first_prompt_gain":gains[idx],
                    "two_training_confirmations":confirms[str(idx)],
                }
        expected_additional=2*len(all_positive)
        expected_diag=6*len(all_positive)
        if (d.get("training_confirmation_model_forwards")!=expected_additional
                or d.get("diagnostic_candidate_model_forwards")!=expected_diag
                or d.get("total_model_loss_forwards")!=17+expected_additional+expected_diag):
            return blocked("incorrect_actual_model_forward_accounting")
        return {
            "schema":SCHEMA,"status":"manual_multiprompt_quality_review_only",
            "seed":seed,"candidate_head_unchanged":True,
            "head_admission_failed":True,
            "selected_model_scope_2048":True,
            "unique_positive_first_prompt_candidates":len(all_positive),
            "additional_train_verification_forwards":expected_additional,
            "total_real_model_loss_forwards":d["total_model_loss_forwards"],
            "real_model_context_forwards":1,
            "single_prompt_selected_candidates_vetoed_by_multi":n_veto,
            "variance_guard_is_heuristic_not_confidence_interval":True,
            "single_prompt_selected_candidates_with_worse_diagnostics":n_harm,
            "results":rows,
            "head_auto_training_permitted":False,
            "independent_execution_attested":False,
            "optimizer_switch_permitted":False,
            "model_update_permitted":False,
            "weight_promotion_permitted":False,
        }
    except (ValueError,KeyError,TypeError,IndexError,OSError,OverflowError,
            RuntimeError,ZeroDivisionError,json.JSONDecodeError):
        return blocked("malformed_or_unreproducible_multiprompt_evidence")

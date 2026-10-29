"""Strict structural review of LFM3 paired 2,048-param B-only gradient arms.

Not proof of execution authenticity, model improvement or equal compute.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path

from src.lfm3_dust_matched_gradient_probe import (
    SCHEMA as SOURCE, ADAMW_LR, MUON_LR, STEPS, TRUST_RADIUS,
)
from src.lfm3_lora_text_task_probe import dataset, _digest
from src.lfm3_real_lora_review import BASE

SCHEMA="sensorgraft.lfm3.matched-lora-b-gradient-review.v1"
H64=re.compile(r"[a-f0-9]{64}\Z")
H32=re.compile(r"[a-f0-9]{32}\Z")


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "trial_authorized":False,"weight_promotion_permitted":False}


def _number(n,lo=0,hi=1e7):
    return type(n) in (float,int) and math.isfinite(n) and lo<=n<=hi


def verify(report):
    if not isinstance(report,dict):
        return blocked("invalid_report")
    expected={
        "schema":SOURCE,"status":"research_parameter_matched_gradient_control",
        "model":"LiquidAI/LFM3-1.7B","base_sha256":BASE,
        "parameter_scope":"last_two_attention_vproj_lora_B_only",
        "selected_parameter_count":2048,"steps":STEPS,
        "actual_backwards":STEPS,"training_forwards":STEPS,
        "selection_score_forwards":36,
        "no_checkpoints_written":True,"final_test_evaluated":False,
        "equal_parameter_scope_with_dust":True,
        "equal_total_forward_or_walltime_budget_with_dust":False,
        "comparable_to_full_qv_lora_optimization":False,
        "source_independently_authenticated":False,
        "trial_authorized":False,"weight_promotion_permitted":False,
    }
    if any(report.get(k)!=v for k,v in expected.items()):
        return blocked("invalid_evidence_or_promotion_status")
    seed,name=report.get("seed"),report.get("optimizer")
    if (type(seed) is not int or seed not in (1,2,3)
            or name not in ("adamw","muon")
            or report.get("learning_rate")!=(ADAMW_LR if name=="adamw" else MUON_LR)
            or report.get("experimental_muon_on_lora_factors") is not (name=="muon")
            or type(report.get("run_id")) is not str
            or not H32.fullmatch(report["run_id"])):
        return blocked("unverified_optimization_config")
    expected_manifests={k:_digest(v) for k,v in dataset(seed).items()}
    if report.get("data_manifest")!=expected_manifests:
        return blocked("nonreproducible_task_data")
    names=report.get("selected_adapter_names")
    if (type(names) is not list or len(names)!=2
            or not all(type(n) is str
                       and n.endswith(".self_attn.v_proj.lora_B.default.weight")
                       for n in names)
            or names[0]==names[1]):
        return blocked("incorrect_parameter_scope")
    for key in ("initial_lora_b_sha256","final_lora_b_sha256"):
        value=report.get(key)
        if type(value) is not str or not H64.fullmatch(value):
            return blocked("malformed_adapter_digest")
    if not _number(report.get("elapsed_training_s"),1e-9,3600):
        return blocked("invalid_walltime")
    scores = (report.get("selection_before"), report.get("selection_after"))
    for score in scores:
        if (not isinstance(score,dict) or type(score.get("correct")) is not int
                or score.get("total")!=9 or not 0<=score["correct"]<=9):
            return blocked("invalid_score_inventory")
    rows=report.get("records")
    if not isinstance(rows,list) or len(rows)!=STEPS:
        return blocked("incomplete_training_telemetry")
    for i,event in enumerate(rows):
        if (not isinstance(event,dict) or set(event)!={
                "step","optimizer","learning_rate","train_loss",
                "raw_grad_norm","clipped_grad_max_norm","update_norm",
                "cumulative_lora_b_update_norm","elapsed_s",
                "actual_forward_backwards"}):
            return blocked("unexpected_step_schema")
        if (type(event["step"]) is not int or event["step"]!=i
                or event["optimizer"]!=name
                or event["learning_rate"]!=report["learning_rate"]
                or event["actual_forward_backwards"]!=1
                or event["clipped_grad_max_norm"]!=1
                or any(not _number(event[k],lo) for k,lo in (
                    ("train_loss",1e-9),("raw_grad_norm",0),
                    ("update_norm",0),("elapsed_s",1e-9)))
                or not _number(event["cumulative_lora_b_update_norm"],
                               0,TRUST_RADIUS+1e-5)):
            return blocked("invalid_or_unbounded_gradient_step")
    return {"schema":SCHEMA,"status":"manual_review_only","seed":seed,
            "optimizer":name,"score_after":scores[1]["correct"],
            "backward_calls":STEPS,"trainable_parameters":2048,
            "data_sha256":expected_manifests,
            "initial_adapter_sha256":report["initial_lora_b_sha256"],
            "parameter_scope_matched_with_dust":True,
            "compute_budget_equalized":False,
            "source_independently_authenticated":False,
            "quality_improvement_proven":False,"trial_authorized":False,
            "weight_promotion_permitted":False}


def matched_pair(adamw,muon,dust):
    x,y=verify(adamw),verify(muon)
    if x["status"]!="manual_review_only" or y["status"]!="manual_review_only":
        return blocked("invalid_gradient_control")
    if (x["optimizer"]!="adamw" or y["optimizer"]!="muon"
            or x["seed"]!=y["seed"] or x["initial_adapter_sha256"]!=y["initial_adapter_sha256"]
            or x["data_sha256"]!=y["data_sha256"]
            or not isinstance(dust,dict) or dust.get("seed")!=x["seed"]
            or dust.get("adapter_policy")!="last_two_attention_vproj_lora_B_only"
            or dust.get("selected_parameter_count")!=2048
            or dust.get("initial_lora_b_sha256")!=x["initial_adapter_sha256"]
            or dust.get("data_manifest")!=x["data_sha256"]
            or dust.get("final_test_evaluated") is not False
            or dust.get("weight_promotion_permitted") is not False):
        return blocked("mismatched_starting_weights_or_parameter_scope")
    return {"schema":SCHEMA,"status":"parameter_scope_matched_manual_review",
            "seed":x["seed"],"adamw":x["score_after"],"muon":y["score_after"],
            "dust":dust["selection_after"]["correct"],
            "equal_parameter_scope":True,"equal_compute":False,
            "optimizer_superiority_proven":False,"trial_authorized":False,
            "weight_promotion_permitted":False}

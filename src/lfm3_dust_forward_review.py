"""Fail-closed structural review of paired forward-only Dust LFM3 pilots.

No report can authenticate itself or authorize training/promotion.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from src.lfm3_lora_text_task_probe import dataset, _digest
from src.lfm3_real_lora_review import BASE

SCHEMA = "sensorgraft.lfm3.dust-forward-review.v1"


def blocked(reason):
    return {"status": "blocked", "schema": SCHEMA, "reason": reason,
            "weight_promotion_permitted": False}


def _finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def verify(report):
    if not isinstance(report, dict):
        return blocked("invalid_report")
    required = {
        "schema": "sensorgraft.lfm3.dust-forward-text-pilot.v1",
        "status": "research_forward_only_text_diagnostic",
        "model": "LiquidAI/LFM3-1.7B", "base_sha256": BASE,
        "adapter_policy": "last_two_attention_vproj_lora_B_only",
        "gradient_backwards": 0, "optimizer_object_used": False,
        "forward_only": True, "final_test_evaluated": False,
        "model_weight_checkpoint_written": False,
        "trial_authorized": False, "weight_promotion_permitted": False,
        "comparable_to_full_lora_adamw_muon": False,
    }
    if any(report.get(k) != v for k, v in required.items()):
        return blocked("invalid_training_or_promotion_claim")
    seed, steps, policy = report.get("seed"), report.get("steps"), report.get("policy")
    if (type(seed) is not int or seed not in (1,2,3)
            or type(steps) is not int or steps != 3
            or policy not in ("attended","random")
            or report.get("selected_parameter_count") != 2048
            or report.get("data_manifest") != {k: _digest(v) for k,v in dataset(seed).items()}
            or not isinstance(report.get("selected_adapter_names"),list)
            or len(report["selected_adapter_names"]) != 2):
        return blocked("inconsistent_seed_or_data")
    for key in ("initial_lora_b_sha256","final_lora_b_sha256"):
        h = report.get(key)
        if type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            return blocked("invalid_adapter_digest")
    if report.get("total_antithetic_forwards") != 12:
        return blocked("unmatched_forward_budget")
    rows = report.get("records")
    if not isinstance(rows,list) or len(rows)!=steps:
        return blocked("incomplete_step_receipts")
    accepted=0
    for step,record in enumerate(rows):
        if not isinstance(record,dict) or record.get("step") != step:
            return blocked("untrusted_step")
        order, selected = record.get("preprobe_order"),record.get("selected_candidates")
        if (not isinstance(order,list) or sorted(order)!=[0,1,2,3]
                or not isinstance(selected,list) or len(selected)!=2
                or (selected!=order[:2] if policy=="attended" else selected!=[0,1])
                or record.get("prior_memory_entries") != 4*step
                or record.get("actual_backwards") != 0
                or record.get("actual_antithetic_forwards") != 4):
            return blocked("noncausal_or_unmatched_probe")
        pairs=record.get("antithetic_probe_results")
        if (not isinstance(pairs,list) or len(pairs)!=2
                or any(not isinstance(x,dict) or x.get("candidate")!=selected[i]
                       or not all(_finite(x.get(k)) for k in ("predicted_gain","plus_gain","minus_gain"))
                       for i,x in enumerate(pairs))):
            return blocked("invalid_probe_evidence")
        b,a,r = (record.get("train_loss_before"),record.get("train_loss_after"),
                 record.get("cumulative_lora_b_update_norm"))
        if (not all(_finite(v) for v in (a,b,r)) or a<=0 or b<=0
                or not 0<=r<=0.15001 or type(record.get("step_accepted")) is not bool
                or (a>b+1e-5 if record["step_accepted"] else abs(a-b)>1e-10)):
            return blocked("violated_update_bound")
        accepted += int(record["step_accepted"])
    if report.get("total_accepted_steps")!=accepted:
        return blocked("mismatched_update_receipts")
    return {"schema":SCHEMA,"status":"manual_review_only","seed":seed,"policy":policy,
            "accepted_steps":accepted,"score_after":report["selection_after"]["correct"],
            "source_authenticated":False,"zero_backward_execution_independently_proven":False,
            "optimizer_superiority_established":False,
            "trial_authorized":False,"weight_promotion_permitted":False}


def pair(attended,random_control):
    a,b=verify(attended),verify(random_control)
    if (a["status"]!="manual_review_only" or b["status"]!="manual_review_only"
            or a["policy"]!="attended" or b["policy"]!="random"
            or a["seed"]!=b["seed"] or a["seed"] not in (1,2,3)
            or attended["initial_lora_b_sha256"]!=random_control["initial_lora_b_sha256"]
            or attended["data_manifest"]!=random_control["data_manifest"]
            or attended["selection_before"]!=random_control["selection_before"]):
        return blocked("unpaired_or_invalid_evidence")
    return {"schema":SCHEMA,"status":"paired_manual_review_only","seed":a["seed"],
            "attended":a,"random_control":b,"independent_improvement_established":False,
            "trial_authorized":False,"weight_promotion_permitted":False}


if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:
        sys.exit("usage: python -m src.lfm3_dust_forward_review ATTENDED.json RANDOM.json")
    files=[Path(x) for x in sys.argv[1:]]
    if any(x.is_symlink() or not x.is_file() or x.stat().st_size>100000 for x in files):
        result=blocked("unsafe_input")
    else:
        result=pair(*(json.loads(x.read_text()) for x in files))
    print(json.dumps(result,sort_keys=True,indent=2))

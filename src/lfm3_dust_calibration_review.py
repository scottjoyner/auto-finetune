"""Fail-closed reviewer for the online-trained two-weight Dust prediction gate.

The HMAC/custody trust gap remains: do not treat unsigned local JSON as
independent proof of training, fairness or causality.
"""
from __future__ import annotations
from copy import deepcopy
import math

from src.lfm3_dust_activation_review import (
    review_activation, compare_activation_to_control,
)

SCHEMA="sensorgraft.lfm3.dust-calibrated-activation-review.v1"


def blocked(reason):
    return {"schema":SCHEMA,"status":"blocked","reason":reason,
            "trial_authorized":False,"weight_promotion_permitted":False}


def _weights(x):
    return (type(x) is list and len(x)==2 and
            all(type(v) in (float,int) and math.isfinite(v)
                and .049999<=v<=.950001 for v in x)
            and abs(sum(x)-1.)<1e-8)


def review(report):
    if (not isinstance(report,dict)
            or report.get("policy")!="activation_calibrated"
            or report.get("direction_head")!="causal_prompt_activation_two_head_train_online_calibrated"):
        return blocked("incorrect_calibrated_policy")
    as_activation=deepcopy(report)
    as_activation["policy"]="activation_attended"
    as_activation["direction_head"]="causal_prompt_activation_two_head_training_probes_only"
    rows=as_activation.get("records")
    if not isinstance(rows,list) or len(rows)!=3:
        return blocked("incomplete_calibration_records")
    previous=0
    gate_changed=False
    for step,row in enumerate(rows):
        if not isinstance(row,dict):
            return blocked("invalid_calibration_row")
        pre=row.pop("calibration_weights_before_probes",None)
        post=row.pop("calibration_weights_after_probes",None)
        before=row.pop("calibration_events_before_probes",None)
        after=row.pop("calibration_events_after_probes",None)
        if (not _weights(pre) or not _weights(post)
                or type(before) is not int or type(after) is not int
                or before!=previous or after<before or after>before+4
                or (step==0 and (before!=0 or after!=0))
                or (step>0 and after!=before+4)):
            return blocked("untrusted_or_noncausal_gate_updates")
        gate_changed=gate_changed or abs(pre[0]-post[0])>1e-10
        previous=after
    base=review_activation(as_activation)
    if base["status"]!="research_review_required":
        return blocked("underlying_activation_contract_denied")
    return {"schema":SCHEMA,"status":"manual_review_only","seed":base["seed"],
            "steps":base["steps"],"selection_accuracy":base["attended_selection_accuracy"],
            "extra_prompt_forwards":base["extra_prompt_only_forwards"],
            "gate_calibration_events":previous,"gate_coefficients_changed":gate_changed,
            "backward_training_reported":False,
            "genuine_head_feedback_adaptation_observed":gate_changed,
            "independent_quality_improvement_proven":False,
            "source_independently_authenticated":False,
            "trial_authorized":False,"weight_promotion_permitted":False}


def compare(calibrated,random_control):
    a=review(calibrated)
    if a["status"]!="manual_review_only":
        return blocked("invalid_calibrated_input")
    transformed=deepcopy(calibrated)
    transformed["policy"]="activation_attended"
    transformed["direction_head"]="causal_prompt_activation_two_head_training_probes_only"
    for row in transformed["records"]:
        for key in ("calibration_weights_before_probes","calibration_weights_after_probes",
                    "calibration_events_before_probes","calibration_events_after_probes"):
            row.pop(key,None)
    b=compare_activation_to_control(transformed,random_control)
    if b["status"]!="paired_manual_review_required":
        return blocked("unmatched_random_control")
    return {"schema":SCHEMA,"status":"paired_manual_review_only",
            "calibrated":a,"random_after":b["control"]["score_after"],
            "locked_test_evaluated":False,
            "head_superiority_proven":False,
            "trial_authorized":False,"weight_promotion_permitted":False}

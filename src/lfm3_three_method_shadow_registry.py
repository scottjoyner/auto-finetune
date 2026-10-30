"""Common read-only LFM3 three-method research interface and admission gate.

AdamW, experimental factor-wise Muon and forward-only predictive Dust share
an EXACT 2,048-scalar last-two-v_proj LoRA-B parameter scope in historical
real-checkpoint trials. This registry validates their *actual* evidence and
never constructs an optimizer, loads a model, writes model weights or offers
a way to override failed prediction-head admission.

The Dust neural prediction head is trained externally via derivative-free
forward evaluations of past gain labels. That does not make it qualified to
choose live update directions: the cross-family results fail the zero-gain
control. Hence all method invocations here are SHADOW-ONLY, including AdamW
and Muon. It cannot authorize production or change existing training gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.lfm3_dust_matched_gradient_review import matched_pair
from src.lfm3_dust_text_transfer_admission import assess

SCHEMA="sensorgraft.lfm3.three-way-shadow-methods.v1"
METHODS=("adamw","muon","predictive_dust")
PARAMETER_SCOPE="last_two_attention_vproj_lora_B_only"
PARAMETER_COUNT=2048
RESEARCH_SEEDS=(1,2,3)
TRANSFER_SEEDS=(10,11,12)
METHOD_SPEC={
    "adamw":{
        "update_mechanism":"autograd_backward_plus_adamw",
        "uses_model_backpropagation":True,
        "implementation":"src.lfm3_dust_matched_gradient_probe",
        "research_optimizer_lr":0.0001,
        "trainable_lora_parameters":PARAMETER_COUNT,
        "requires_trained_prediction_head":False,
    },
    "muon":{
        "update_mechanism":"autograd_backward_plus_experimental_factorwise_muon",
        "uses_model_backpropagation":True,
        "implementation":"src.lfm3_dust_matched_gradient_probe",
        "research_optimizer_lr":0.00016,
        "trainable_lora_parameters":PARAMETER_COUNT,
        "requires_trained_prediction_head":False,
    },
    "predictive_dust":{
        "update_mechanism":"antithetic_forward_only_lora_b_probe",
        "uses_model_backpropagation":False,
        "implementation":"src.lfm3_dust_predictive_probe",
        "head_implementation":"src.lfm3_dust_trainable_attention",
        "trainable_lora_parameters":PARAMETER_COUNT,
        "external_neural_head_parameters":195,
        "requires_trained_prediction_head":True,
        "head_status":"cross_family_transfer_hold_not_qualified",
    },
}
MAX_FILE_BYTES=175000


def blocked(reason):
    return {
        "schema":SCHEMA,"status":"blocked","reason":reason,
        "optimizer_switch_permitted":False,"model_training_permitted":False,
        "weight_promotion_permitted":False,"production_execution_permitted":False,
    }


def _read(root,relative):
    """Read only fixed, checked-in source filenames, never arbitrary inputs."""
    p=root.joinpath(*relative.split("/"))
    if (p.is_symlink() or not p.is_file()
            or p.stat().st_size<100 or p.stat().st_size>MAX_FILE_BYTES):
        raise ValueError("missing_or_oversized_research_evidence")
    data=p.read_bytes()
    payload=json.loads(data)
    if not isinstance(payload,dict):
        raise ValueError("malformed_research_evidence")
    return payload,hashlib.sha256(data).hexdigest()


def evidence_paths():
    paths={}
    for s in RESEARCH_SEEDS:
        for method in ("adamw","muon"):
            paths[f"matched_seed{s}_{method}"]=(
                f"research/evidence/lfm3-dust-matched-three-way-20261029/"
                f"seed{s}-{method}-matched-b3.json")
        paths[f"matched_seed{s}_dust"]=(
            "research/evidence/lfm3-dust-forward-20261029/"
            f"seed{s}-random-3steps.json")
    for s in TRANSFER_SEEDS:
        paths[f"transfer_seed{s}"]=(
            "research/evidence/lfm3-dust-text-transfer-20261029/"
            f"seed{s}-extraction-to-uppercase-18episodes.json")
    return paths


def review(root,*,requested_method="all",intent="inspect"):
    if type(requested_method) is not str or requested_method not in ("all",)+METHODS:
        return blocked("unknown_optimizer_method")
    if type(intent) is not str or intent!="inspect":
        return blocked("this_interface_supports_only_read_only_research_inspection")
    if not isinstance(root,(str,Path)):
        return blocked("invalid_repository_path")
    folder=Path(root)
    if not folder.is_dir() or folder.is_symlink():
        return blocked("missing_or_symlink_repository_root")
    try:
        data={}
        digests={}
        for key,path in evidence_paths().items():
            data[key],digests[key]=_read(folder,path)
        matched=[]
        for seed in RESEARCH_SEEDS:
            x=matched_pair(
                data[f"matched_seed{seed}_adamw"],
                data[f"matched_seed{seed}_muon"],
                data[f"matched_seed{seed}_dust"])
            if (x.get("status")!="parameter_scope_matched_manual_review"
                    or x.get("equal_parameter_scope") is not True
                    or x.get("equal_compute") is not False
                    or x.get("weight_promotion_permitted") is not False):
                return blocked(f"unverified_matching_adapter_seed_{seed}")
            matched.append(x)
        transfer=assess([data[f"transfer_seed{s}"] for s in TRANSFER_SEEDS])
        if (transfer.get("status")!="hold_not_qualified"
                or transfer.get("predictor_qualified_for_model_updates") is not False
                or transfer.get("model_update_permitted") is not False
                or transfer.get("weight_promotion_permitted") is not False):
            # Fail closed on surprising changed evidence (including a future
            # passing report); a new approval design must be reviewed first.
            return blocked("unexpected_or_invalid_neural_head_admission")
        if any(m.get("adamw")!=m.get("muon") or m.get("adamw")!=m.get("dust")
               for m in matched):
            return blocked("inconsistent_three_way_task_score_inventory")
        methods={}
        for name in METHODS:
            methods[name]={
                **METHOD_SPEC[name],
                "name":name,"research_only":True,
                "implementation_executable_by_this_registry":False,
                "model_training_permitted":False,
                "optimizer_switch_permitted":False,
                "weight_promotion_permitted":False,
                "same_pretrained_initial_adapter":True,
                "same_2048_parameter_scope":True,
                "matched_total_compute":False,
                "previous_experiment_selection_scores":{
                    str(entry["seed"]):entry[
                        "dust" if name=="predictive_dust" else name]
                    for entry in matched
                },
                "head_recommendations_enabled":False,
                "reason":"cross_family_predictive_head_failed_zero_gain_baseline"
                         if name=="predictive_dust"
                         else "research_pilot_only_requires_separate_release_review",
            }
        return {
            "schema":SCHEMA,"status":"verified_shadow_only",
            "requested_method":requested_method,"intent":"inspect",
            "base_model":"LiquidAI/LFM3-1.7B",
            "model_adapter_parameter_scope":PARAMETER_SCOPE,
            "model_adapter_parameters":PARAMETER_COUNT,
            "source_receipts_independently_authenticated":False,
            "matched_parameter_scope":True,"matched_total_compute":False,
            "historical_three_seed_selection_accuracy":{
                str(x["seed"]):x["adamw"] for x in matched
            },
            "neural_head_transfer_admission":transfer["status"],
            "neural_head_transfer_seedwise_mse_ratio":{
                str(x["seed"]):x["neural_mse_ratio"]
                for x in (transfer["per_seed"][str(s)] for s in TRANSFER_SEEDS)
            },
            "neural_head_transfer_blockers":transfer["blocking_reasons"],
            "methods":methods if requested_method=="all" else {
                requested_method:methods[requested_method]},
            "evidence_sha256":digests,
            "source_evidence_is_local_unsigned_json":True,
            "trials_can_be_executed_by_this_interface":False,
            "model_training_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
            "production_execution_permitted":False,
        }
    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError,
            UnicodeError,json.JSONDecodeError):
        return blocked("invalid_or_unreadable_research_evidence")


def shadow_rank(head,context,directions,*,episode,admission):
    """Read-only connection to actual neural head; NEVER actuates model update.

    Even with a favorable head score, research admission must be separately
    satisfied and production controls stay false. Here the known evidence is
    on HOLD. No gradients or model/tensor objects are created in this method.
    """
    if (not isinstance(admission,dict)
            or admission.get("status")!="hold_not_qualified"
            or admission.get("predictor_qualified_for_model_updates") is not False
            or not isinstance(directions,list) or len(directions)!=8
            or type(episode) is not int or episode<0):
        return blocked("invalid_shadow_head_or_admission")
    scored=[]
    for i,coordinates in enumerate(directions):
        forecast=head.predict(coordinates,context,episode=episode)
        gain=forecast["predicted_gain"]
        if type(gain) not in (int,float):
            return blocked("bad_head_prediction")
        import math
        if not math.isfinite(gain):
            return blocked("nonfinite_head_prediction")
        scored.append({"candidate_index":i,"predicted_gain":gain,
                       "would_actuate":False})
    return {
        "schema":SCHEMA,"status":"predictive_head_shadow_abstention",
        "candidate_order_high_to_low":[x["candidate_index"] for x in sorted(
            scored,key=lambda x:(-x["predicted_gain"],x["candidate_index"]))],
        "candidates":scored,
        "selected_for_live_probe":None,
        "admission":"hold_not_qualified",
        "model_training_permitted":False,
        "optimizer_switch_permitted":False,
        "weight_promotion_permitted":False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-only",action="store_true",required=True)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--method",choices=("all",)+METHODS,default="all")
    parser.add_argument("--intent",default="inspect")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=review(args.root,requested_method=args.method,intent=args.intent)
    if args.output:
        with args.output.open("x",encoding="utf-8") as f:
            json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    if result["status"]=="blocked":
        raise SystemExit(2)


if __name__=="__main__":
    main()

"""Aggregate, non-behavioral fidelity audit of one frozen 3-arm LFM3 study.

Reports the existing edited checkpoint's mechanical integrity and previously
stored heldout activation projections. Does not train, edit, generate,
publish model weights, infer refusal efficacy, or assess response safety.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from src.build_abliterated_checkpoint import DEFAULT_OUTPUT, DEFAULT_SOURCE, DEFAULT_VECTOR, digest
from src.verify_checkpoint import verify


def summarize(provenance, prior, verification):
    if provenance.get("method") != "M1-style row-space weight projection":
        raise ValueError("unexpected edit method")
    if provenance.get("release_authorized") is not False or provenance.get("publication_approved") is not False:
        raise ValueError("release authorization state must stay closed")
    if provenance.get("edited_matrices") != 48 or provenance.get("direction_layer") != 10 or provenance.get("alpha") != 1.0:
        raise ValueError("frozen M1 experiment not identified")
    if provenance.get("source_sha256") != verification.get("source_sha256") or \
       provenance.get("checkpoint_sha256") != verification.get("target_sha256"):
        raise ValueError("verification/model provenance mismatch")
    if verification.get("exact_projection_verified") is not True or \
       verification.get("source_tensor_count") != 286 or \
       verification.get("selected_residual_writers") != 48 or \
       verification.get("actual_changed_tensors") != 48 or \
       verification.get("verified_non_target_unchanged") != 238:
        raise ValueError("full tensor-by-tensor projection identity not proven")
    if prior.get("original_checkpoint_sha256") != provenance["source_sha256"] or \
       prior.get("edited_checkpoint_sha256") != provenance["checkpoint_sha256"]:
        raise ValueError("private activation audit from a different model")
    fitted=prior.get("calibration_case_ids")
    holdout=prior.get("holdout_case_ids")
    if not isinstance(fitted,list) or len(fitted)!=2 or len(set(fitted))!=2:
        raise ValueError("two frozen calibration case IDs required")
    if not isinstance(holdout,list) or len(holdout)!=4 or len(set(holdout))!=4 or set(fitted)&set(holdout):
        raise ValueError("requires four disjoint frozen activation-audit prompts")
    ratios=prior.get("ratios_edited_to_original_by_layer")
    if not isinstance(ratios,dict) or set(ratios)!={str(i) for i in range(1,24)}:
        raise ValueError("expected the original 23 nonzero contrast measurement layers")
    vals=[]
    for r in ratios.values():
        if type(r) not in (int,float) or not math.isfinite(r) or r<0:
            raise ValueError("invalid heldout projection ratio")
        vals.append(float(r))
    reduced=sum(v<1 for v in vals)
    increased=sum(v>1 for v in vals)
    return {
        "evidence_class":"non-behavioral structural and heldout activation-direction fidelity",
        "scope":"pre-existing A original and B M1-abliterated in fixed A/B/C study",
        "edited_checkpoint": "B_M1_abliterated_smoke",
        "source_sha256":provenance["source_sha256"],
        "edited_sha256":provenance["checkpoint_sha256"],
        "direction_sha256":provenance["direction_sha256"],
        "direction_source_layer":10,
        "strength_alpha":1.0,
        "original_tensor_count":286,
        "exactly_edited_residual_writing_matrices":48,
        "other_tensors_verified_unchanged":238,
        "exact_projection_verified":True,
        "calibration_pairs":2,
        "disjoint_holdout_prompt_pairs":4,
        "activation_layers_measured":23,
        "heldout_projection_ratio_median":round(statistics.median(vals),6),
        "heldout_projection_ratio_min":round(min(vals),6),
        "heldout_projection_ratio_max":round(max(vals),6),
        "measured_layers_with_reduced_projection":reduced,
        "measured_layers_with_increased_projection":increased,
        "refusal_effectiveness_validated":False,
        "human_safety_validated":False,
        "release_approved":False,
        "limitations":[
            "Arditi-style rank-one weight projection is present, not a full reproduction of published candidate selection",
            "The refusal direction was estimated from just two calibration prompt pairs",
            "No multi-layer direction candidate validation or robust calibration-set uncertainty analysis",
            "LFM3 sandwich normalization after edited writers can restore components along this direction",
            "Four heldout prompt pairs constrain projection measurements; not a behavioral efficacy test",
            "Projection reductions do not imply refusal removal or safe model responses",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--provenance",type=Path,
                   default=DEFAULT_OUTPUT/"RESEARCH_PROVENANCE.json")
    p.add_argument("--private-activation-audit",type=Path,
                   default=Path(__file__).resolve().parents[1]/"results/abliteration_fidelity_private.json")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace historical research evidence")
    # Performs SHA-256 and tensor-for-tensor checks on the actual frozen
    # original and modified local checkpoint, with no changed weights.
    verification=verify()
    if digest(DEFAULT_VECTOR)!=json.loads(args.provenance.read_text())["direction_sha256"]:
        raise ValueError("wrong frozen direction vector")
    data=summarize(json.loads(args.provenance.read_text()),
                   json.loads(args.private_activation_audit.read_text()),
                   verification)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()

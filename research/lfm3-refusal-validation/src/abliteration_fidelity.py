"""Check actual residual-direction effects from M1 editing, without model generation.

Uses disjoint, private prompt contrasts. Report projection energy, not safety.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

from src.refusal_probe import encode, load_local_model, hidden_by_layer
from src.eval_contracts import read_jsonl, dataset_sha256
from src.three_checkpoint_eval import ARMS, checked_arm_paths

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "paired_v0.jsonl"
DIRECTIONS = ROOT / "results" / "directions.safetensors"


def projection_report(captures, ref_direction):
    """RMS projected component, overall RMS, and normalized fraction by layer."""
    report = {}
    for index, values in captures.items():
        unit = ref_direction[f"layer_{index}"].float()
        vec = torch.stack(values).float()
        dot = vec @ unit
        component_rms = torch.sqrt(dot.square().mean()).item()
        full_rms = torch.sqrt(vec.square().sum(dim=-1).mean()).item()
        report[str(index)] = {
            "projected_rms": round(component_rms, 8),
            "hidden_rms": round(full_rms, 8),
            "fraction_of_hidden_rms": round(
                component_rms / full_rms if full_rms else 0.0, 8
            )
        }
    return report


def verify_holdout(pairs, fitting_ids):
    selected = [p for p in pairs if p["id"] not in fitting_ids]
    if not selected:
        raise ValueError("holdout set empty")
    if set(x["id"] for x in selected) & set(fitting_ids):
        raise ValueError("holdout contamination")
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-holdout-pairs", type=int, default=2)
    p.add_argument("--output", type=Path,
                   default=ROOT / "results/abliteration_fidelity_private.json")
    args = p.parse_args()
    os.umask(0o077)
    models = checked_arm_paths(ARMS)
    ref = load_file(str(DIRECTIONS), device="cpu")
    fitting = ["p01", "p02"]
    heldout = verify_holdout(read_jsonl(PAIRS), fitting)
    if not 1 <= args.max_holdout_pairs <= len(heldout):
        p.error("invalid holdout count")
    selected = heldout[:args.max_holdout_pairs]
    # Layer zero intentionally omitted where its calibration contrast is zero.
    measurements = {}
    for name in ("A_original_posttrained", "B_M1_abliterated_smoke"):
        model, tok = load_local_model(ARMS[name])
        capture = {i:[] for i in range(24) if f"layer_{i}" in ref}
        for pair in selected:
            for variant in ("benign", "adversarial"):
                hid = hidden_by_layer(model, encode(tok, pair[variant]))
                for idx in capture:
                    capture[idx].append(hid[idx])
        measurements[name] = projection_report(capture, ref)
        print(f"{name}: captured {len(selected)*2} prompts", flush=True)
        del model, tok
        gc.collect()
    ratio = {}
    for index in measurements["A_original_posttrained"]:
        b = measurements["B_M1_abliterated_smoke"][index]["projected_rms"]
        a = measurements["A_original_posttrained"][index]["projected_rms"]
        ratio[index] = round(b/a, 6) if a > 1e-9 else None
    result = {
        "study":"M1-style weight edit activation-fidelity audit",
        "data_sha256":dataset_sha256(PAIRS),
        "calibration_case_ids":fitting,
        "holdout_case_ids":[x["id"] for x in selected],
        "original_checkpoint_sha256":models["A_original_posttrained"],
        "edited_checkpoint_sha256":models["B_M1_abliterated_smoke"],
        "ratios_edited_to_original_by_layer":ratio,
        "detailed_private_layers":measurements,
        "warning":"Activation direction is only a two-pair contrast. Ratios are NOT refusal or safety efficacy.",
    }
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    os.chmod(args.output,0o600)
    print(json.dumps({
        "holdout_case_ids":result["holdout_case_ids"],
        "ratios_edited_to_original_by_layer":ratio,
        "warning":result["warning"]},indent=2))


if __name__ == "__main__":
    main()

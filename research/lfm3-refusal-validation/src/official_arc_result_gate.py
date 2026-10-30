"""Strict import gate for independently executed lm-eval 0.4.13 ARC reports.

No benchmark is executed here; accepts only three complete, model-identity-
pinned official result files. Passing this structural gate is NOT proof that
a third-party run actually occurred or that reported outcomes are accurate.
"""
import argparse
import json
import math
import os
from pathlib import Path

from src.official_arc_readiness import ARMS_EXPECTED, EXPECTED_CASES, HARNESS_VERSION
from src.three_checkpoint_eval import ARMS, checked_arm_paths


def _pretrained(model_args):
    if isinstance(model_args, dict):
        value=model_args.get("pretrained")
    elif isinstance(model_args, str):
        params=dict(part.split("=",1) for part in model_args.split(",") if "=" in part)
        value=params.get("pretrained")
    else:
        raise ValueError("missing model_args")
    if not isinstance(value,str) or not value.strip():
        raise ValueError("missing pretrained model path")
    return Path(value).resolve(strict=True)


def check_one(arm, data, static):
    if arm not in ARMS_EXPECTED:
        raise ValueError("unexpected model arm")
    if data.get("config",{}).get("model")!="hf":
        raise ValueError("not the pinned huggingface evaluator")
    config=data["config"]
    if config.get("limit") is not None:
        raise ValueError("limited or sampled benchmark results are not full ARC")
    if _pretrained(config.get("model_args"))!=ARMS[arm].resolve(strict=True):
        raise ValueError("model path does not identify the frozen arm")
    if data.get("n-shot",{}).get("arc_challenge")!=0:
        raise ValueError("not zero-shot ARC")
    task=data.get("configs",{}).get("arc_challenge",{})
    for key, value in (
        ("dataset_path","allenai/ai2_arc"),
        ("dataset_name","ARC-Challenge"),
        ("output_type","multiple_choice"),
        ("test_split","test"),
    ):
        if task.get(key)!=value:
            raise ValueError("official ARC task config mismatch")
    counts=data.get("n-samples",{}).get("arc_challenge",{})
    if counts.get("original")!=EXPECTED_CASES or counts.get("effective")!=EXPECTED_CASES:
        raise ValueError("not full 1172-case ARC scoring")
    row=data.get("results",{}).get("arc_challenge",{})
    if row.get("sample_len")!=EXPECTED_CASES:
        raise ValueError("reported sample length mismatch")
    out={}
    for label in ("acc","acc_norm"):
        value=row.get(f"{label},none")
        stderr=row.get(f"{label}_stderr,none")
        if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f"missing or invalid official {label}")
        if type(stderr) not in (float,int) or not math.isfinite(stderr) or not 0<=stderr<=1:
            raise ValueError(f"missing or invalid official {label} standard error")
        out[label]=float(value)
        out[label+"_stderr"]=float(stderr)
    return out


def check_all(official_runs, static, hashes):
    if static.get("kind")!="static readiness only: no model inference" or \
       static.get("harness",{}).get("version")!=HARNESS_VERSION or \
       static.get("dataset",{}).get("n_unique_questions")!=EXPECTED_CASES or \
       static.get("checkpoint_sha256")!=hashes or \
       static.get("checkpoint_count")!=3:
        raise ValueError("static contract or frozen model hashes changed")
    if tuple(official_runs)!=ARMS_EXPECTED:
        raise ValueError("expected exactly the three original arms in order")
    reports={arm:check_one(arm,official_runs[arm],static) for arm in ARMS_EXPECTED}
    return {
        "evidence_class":"external lm-eval aggregate results structural validation only",
        "benchmark":"EleutherAI arc_challenge",
        "harness_version_required":HARNESS_VERSION,
        "n_questions_per_arm":EXPECTED_CASES,
        "checkpoint_count":3,
        "checkpoint_sha256":hashes,
        "reported_metrics":reports,
        "paired_case_level_significance_calculated":False,
        "independent_run_provenance_verified":False,
        "human_safety_review_complete":False,
        "zero_refusal_supported":False,
        "release_approved":False,
        "limitations":[
            "Validation of JSON metadata cannot authenticate a benchmark execution",
            "Require independent compute logs, actual task dataset hash and local custody receipt",
            "Aggregate metrics do not establish matched-question effects without case-level audit",
            "Custom likelihood scores are not substituted for official acc/acc_norm",
            "A three-arm benchmark does not replace independent human refusal/safety review",
        ],
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--static-readiness",type=Path,required=True)
    for arm in ARMS_EXPECTED:
        parser.add_argument("--"+arm.replace("_","-"),dest=arm,type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing research report")
    static=json.loads(args.static_readiness.read_text())
    runs={}
    for arm in ARMS_EXPECTED:
        path=getattr(args,arm)
        if path.is_symlink() or not path.is_file():
            raise ValueError("result must be a regular local file")
        runs[arm]=json.loads(path.read_text())
    report=check_all(runs,static,checked_arm_paths(ARMS))
    os.umask(0o077)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(report,indent=2)+"\n")
    args.output.chmod(0o600)
    print(json.dumps({
        "static_result_structure_validated":True,
        "independent_execution_provenance_verified":False,
        "model_release_approved":False,
        "report":str(args.output),
    },indent=2))


if __name__=="__main__":
    main()

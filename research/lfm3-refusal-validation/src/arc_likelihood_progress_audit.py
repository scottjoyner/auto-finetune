"""Read-only ARC-Challenge A/B/C journal integrity and completion audit.

Never scores model safety, modifies a journal, or exports question text.
This audits the custom full-choice-text likelihood benchmark, not an official
lm-evaluation-harness benchmark run.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from src.arc_capability_eval import read_arc, SOURCE_SHA256, SOURCE_REVISION
from src.arc_full_likelihood_eval import prepare, manifest_for, ORDER
from src.three_checkpoint_eval import ARMS
from src.xstest_progress_audit import snapshot_complete_lines


def validate_manifest(manifest, cases, model_hashes=None):
    if manifest["dataset_sha256"] != SOURCE_SHA256:
        raise ValueError("dataset checksum mismatch")
    if manifest["revision"] != SOURCE_REVISION:
        raise ValueError("dataset revision mismatch")
    if manifest["checkpoint_order"] != list(ORDER) or len(ORDER) != 3:
        raise ValueError("exactly three fixed A/B/C checkpoints required")
    if len(set(manifest["checkpoint_sha256"].values())) != 3:
        raise ValueError("checkpoint SHA-256 digests are not distinct")
    if set(manifest["checkpoint_sha256"]) != set(ORDER):
        raise ValueError("checkpoint list mismatch")
    if model_hashes is not None and manifest["checkpoint_sha256"] != model_hashes:
        raise ValueError("local checkpoint weights differ from journal manifest")
    expected = manifest_for(cases, manifest["checkpoint_sha256"], len(cases))
    if manifest != expected:
        raise ValueError("ARC manifest differs from frozen source and settings")
    return True


def validate_journal(manifest, rows, cases):
    expected = [(arm, case) for arm in ORDER for case in cases]
    if len(rows) > len(expected):
        raise ValueError("extra journal rows")
    for idx, row in enumerate(rows):
        arm, case = expected[idx]
        if row.get("arm") != arm or row.get("id") != case["id"]:
            raise ValueError("ARC journal is not a contiguous ordered prefix")
        count=len(case["options"])
        correct=case["correct_index"]
        if row.get("correct_index") != correct:
            raise ValueError("ARC correct answer changed")
        if not isinstance(row.get("candidate_token_lengths"), list) or len(row["candidate_token_lengths"]) != count:
            raise ValueError("invalid candidate token lengths")
        if any(type(x) is not int or not 1 <= x <= 512 for x in row["candidate_token_lengths"]):
            raise ValueError("invalid candidate token lengths")
        for field in ("raw_logprobs", "mean_per_token_logprobs"):
            values=row.get(field)
            if not isinstance(values,list) or len(values)!=count:
                raise ValueError("missing candidate log-probabilities")
            if any(type(v) not in (int,float) or not math.isfinite(v) or v > 0 for v in values):
                raise ValueError("invalid finite conditional log-probability")
        for pred_col, score_col, correct_col in (
            ("predicted_raw_index", "raw_logprobs", "raw_correct"),
            ("predicted_length_normalized_index", "mean_per_token_logprobs", "length_normalized_correct")
        ):
            predicted=row.get(pred_col)
            if type(predicted) is not int or not 0<=predicted<count:
                raise ValueError("invalid choice prediction index")
            if predicted != max(range(count),key=lambda i:row[score_col][i]):
                raise ValueError("prediction inconsistent with recorded scores")
            if type(row.get(correct_col)) is not bool or row[correct_col] != (predicted==correct):
                raise ValueError("correctness field inconsistent with expected answer")
        elapsed=row.get("elapsed_seconds")
        if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid elapsed time")
    return len(rows) == len(expected)


def progress_summary(manifest, rows, complete, partial, digest):
    n=manifest["sample_size"]
    counts=Counter(r["arm"] for r in rows)
    return {
        "benchmark":"ARC-Challenge, custom answer-text conditional likelihood",
        "journal_integrity":"passed",
        "model_arms":list(ORDER),
        "expected_cases":n*len(ORDER),
        "recorded_cases":len(rows),
        "completion": "generations_complete" if complete and not partial else "in_progress",
        "per_arm":{arm:{"recorded":counts[arm],"remaining":n-counts[arm]} for arm in ORDER},
        "journal_complete_lines_sha256":digest,
        "unterminated_tail_present":partial,
        "benchmark_source_sha256":SOURCE_SHA256,
        "checkpoint_sha256":manifest["checkpoint_sha256"],
        "safety_outcomes_validated":False,
        "official_arc_score":False,
        "limitations": [
            "This custom likelihood scorer is not an official benchmark implementation",
            "Human grading of XSTest malicious-action resistance is independent and pending",
            "Avoid selective interpretation of partial test-run scores"
        ],
    }


def audit(results_dir, verify_live_weights=False):
    results_dir=Path(results_dir)
    manifest=json.loads((results_dir/"private_manifest.json").read_text())
    all_cases=[prepare(c) for c in read_arc()]
    if manifest.get("sample_size") == 1172:
        chosen=all_cases
    else:
        import random
        size=manifest.get("sample_size")
        if type(size) is not int or not 1<=size<=1172:
            raise ValueError("unsupported ARC sample size")
        chosen=random.Random(17).sample(all_cases,size)
    live_hashes=None
    if verify_live_weights:
        from src.three_checkpoint_eval import checked_arm_paths
        live_hashes=checked_arm_paths(ARMS)
    validate_manifest(manifest,chosen,live_hashes)
    rows,partial,digest=snapshot_complete_lines(results_dir/"private_cases.jsonl")
    complete=validate_journal(manifest,rows,chosen)
    return progress_summary(manifest,rows,complete,partial,digest)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--verify-live-weights",action="store_true")
    args=parser.parse_args()
    print(json.dumps(audit(args.results_dir,args.verify_live_weights),indent=2))


if __name__=="__main__":
    main()

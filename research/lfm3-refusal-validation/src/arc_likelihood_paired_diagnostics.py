"""Paired three-checkpoint ARC-Challenge custom-likelihood diagnostics.

Provides public aggregate provenance and paired bootstrap intervals only.
Raw answer keys, questions, options and individual model predictions stay private.
This is NOT an official lm-evaluation-harness leaderboard result.
"""
import argparse
import json
from pathlib import Path

from src.arc_likelihood_progress_audit import audit
from src.xstest_progress_audit import snapshot_complete_lines
from src.arc_full_likelihood_eval import ORDER
from src.paired_reviewed_effects import paired_interval
from src.reviewed_xstest_metrics import wilson_interval


METRICS = (
    ("length_normalized_correct", "predicted_length_normalized_index"),
    ("raw_correct", "predicted_raw_index"),
)


def paired_diagnostics(records, expected_per_arm=1172, seed=17, resamples=2000):
    if expected_per_arm != 1172 or len(records) != expected_per_arm * len(ORDER):
        raise ValueError("full 1,172-case, exactly three-arm evaluation required")
    by_model = {arm:{} for arm in ORDER}
    for row in records:
        arm, case_id = row.get("arm"), row.get("id")
        if arm not in by_model or case_id in by_model[arm]:
            raise ValueError("duplicate or unknown checkpoint/model-case record")
        by_model[arm][case_id] = row
    ids = set(by_model[ORDER[0]])
    if len(ids) != expected_per_arm or any(set(by_model[arm]) != ids for arm in ORDER):
        raise ValueError("unmatched case sets across original/edited/base")
    if any(len({by_model[arm][key]["correct_index"] for arm in ORDER}) != 1 for key in ids):
        raise ValueError("answer key mismatch across checkpoint scores")
    sorted_ids = sorted(ids)
    result = {
        "study":"ARC-Challenge exact same question A/B/C custom likelihood",
        "evidence_class":"descriptive capability proxy, not official ARC score",
        "cases_per_model":expected_per_arm,
        "total_scored":len(records),
        "safety_evidence":False,
        "metrics":{},
    }
    for metric, pred in METRICS:
        counts = {}
        for arm in ORDER:
            correct = sum(by_model[arm][id][metric] for id in sorted_ids)
            low, high = wilson_interval(correct, len(sorted_ids))
            counts[arm] = {
                "correct":correct,
                "total":len(sorted_ids),
                "accuracy":round(correct/len(sorted_ids),6),
                "wilson_95pct":[round(low,6),round(high,6)],
            }
        comparisons = {}
        for left, right in ((ORDER[0], ORDER[1]), (ORDER[0], ORDER[2]), (ORDER[1], ORDER[2])):
            gain = sum(not by_model[left][id][metric] and by_model[right][id][metric] for id in sorted_ids)
            loss = sum(by_model[left][id][metric] and not by_model[right][id][metric] for id in sorted_ids)
            same_option = sum(
                by_model[left][id][pred] == by_model[right][id][pred]
                for id in sorted_ids
            )
            diffs = [
                int(by_model[right][id][metric]) - int(by_model[left][id][metric])
                for id in sorted_ids
            ]
            interval = paired_interval(diffs,seed=seed,resamples=resamples)
            if gain - loss != sum(diffs):
                raise ValueError("inconsistent matched capability flip counts")
            comparisons[f"{right}_minus_{left}"] = {
                "became_correct":gain,
                "became_incorrect":loss,
                "same_chosen_option":same_option,
                "paired_accuracy_delta":interval,
            }
        result["metrics"][metric] = {
            "by_checkpoint":counts,
            "paired_comparisons":comparisons,
        }
    result["limitations"] = [
        "This is a custom raw-completion, answer-text conditional-likelihood protocol",
        "Not official ARC benchmark scoring or an instruction-following evaluation",
        "Single answer-option ordering and zero-shot prompt format; pretrained model may be favored",
        "Wilson intervals summarize per-model binomial variation; paired intervals resample matched questions",
        "Edited model refusal direction fitted from only two calibration pairs",
        "Model scores cannot establish validated refusal or malicious-action resistance",
    ]
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--verify-live-weights",action="store_true")
    args=p.parse_args()
    state=audit(args.results_dir,verify_live_weights=args.verify_live_weights)
    if state["completion"]!="generations_complete" or state["recorded_cases"]!=3516:
        raise ValueError("refusing to interpret incomplete capability run")
    records,partial,journal_hash = snapshot_complete_lines(args.results_dir/"private_cases.jsonl")
    if partial:
        raise ValueError("unterminated journal tail")
    data=paired_diagnostics(records)
    data["benchmark_sha256"]=state["benchmark_source_sha256"]
    data["checkpoint_sha256"]=state["checkpoint_sha256"]
    data["journal_sha256"]=journal_hash
    if args.output.exists():
        raise FileExistsError("refusing to overwrite published diagnostic evidence")
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()

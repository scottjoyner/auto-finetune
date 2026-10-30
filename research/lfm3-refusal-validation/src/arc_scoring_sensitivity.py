"""Read-only scoring-format sensitivity from the completed three-arm ARC run.

Produces aggregate diagnostics only. No case text, answer IDs, predictions,
model changes, network calls, new evaluation prompts, or safety claims.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.arc_full_likelihood_eval import ORDER
from src.arc_likelihood_progress_audit import audit
from src.xstest_progress_audit import snapshot_complete_lines


def compare_scoring(records, n_per_arm=1172):
    if n_per_arm != 1172 or len(records) != n_per_arm * len(ORDER):
        raise ValueError("requires 1,172 complete rows from each frozen checkpoint")
    groups={arm:{} for arm in ORDER}
    for r in records:
        arm=r.get("arm")
        if arm not in groups or r.get("id") in groups[arm]:
            raise ValueError("duplicate or unrecognized ARC item")
        groups[arm][r["id"]]=r
    ids=set(groups[ORDER[0]])
    if len(ids) != n_per_arm or any(set(groups[arm])!=ids for arm in ORDER):
        raise ValueError("case sets not paired across the 3 model arms")
    counts=Counter()
    expected_random=0.0
    unique_shortest_cases=0
    for case_id in ids:
        a=groups[ORDER[0]][case_id]
        lengths=a["candidate_token_lengths"]
        if len(lengths) not in (2,3,4,5) or not all(isinstance(v,int) and v>0 for v in lengths):
            raise ValueError("invalid answer-choice tokenization")
        if any(groups[arm][case_id]["candidate_token_lengths"] != lengths for arm in ORDER):
            raise ValueError("tokenization differs across checkpoint arms")
        count=len(lengths)
        counts[count]+=1
        expected_random += 1.0/count
        if lengths.count(min(lengths))==1:
            unique_shortest_cases += 1
    by_arm={}
    for arm in ORDER:
        raw_vs_normalized_changed=0
        norm_rescued=0
        norm_harmed=0
        raw_shortest_unique=0
        norm_shortest_unique=0
        same_correctness=0
        for case_id in ids:
            item=groups[arm][case_id]
            raw=item["predicted_raw_index"]
            normalized=item["predicted_length_normalized_index"]
            r_correct=item["raw_correct"]
            n_correct=item["length_normalized_correct"]
            if type(r_correct) is not bool or type(n_correct) is not bool:
                raise ValueError("boolean correctness expected")
            raw_vs_normalized_changed+=(raw!=normalized)
            norm_rescued+=(not r_correct and n_correct)
            norm_harmed+=(r_correct and not n_correct)
            same_correctness+=(r_correct==n_correct)
            lengths=item["candidate_token_lengths"]
            if lengths.count(min(lengths)) == 1:
                shortest=lengths.index(min(lengths))
                raw_shortest_unique+=(raw==shortest)
                norm_shortest_unique+=(normalized==shortest)
        by_arm[arm]={
            "raw_vs_normalized_different_choice":raw_vs_normalized_changed,
            "normalized_rescued_correct":norm_rescued,
            "normalized_lost_correct":norm_harmed,
            "same_correctness_under_both_scoring_rules":same_correctness,
            "unique_shortest_selected_raw":raw_shortest_unique,
            "unique_shortest_selected_normalized":norm_shortest_unique,
            "unique_shortest_case_count":unique_shortest_cases,
        }
    return {
        "study":"completed three-arm ARC-Challenge custom-likelihood sensitivity audit",
        "number_of_checkpoints":len(ORDER),
        "questions":n_per_arm,
        "question_option_count_distribution":{str(k):v for k,v in sorted(counts.items())},
        "uniform_random_option_expected_accuracy":round(expected_random/n_per_arm,6),
        "unique_shortest_option_cases":unique_shortest_cases,
        "by_arm":by_arm,
        "safety_validated":False,
        "official_arc_scoring":False,
        "cautions":[
            "Different likelihood aggregation rules can change which answer appears best",
            "Length-normalization is an analysis convention, not an official ARC score",
            "The unique shortest-answer statistic does not include ties",
            "One answer ordering and prompt format; these results do not validate model safety",
        ]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    state=audit(args.results_dir)
    if state["completion"]!="generations_complete" or state["recorded_cases"]!=3516:
        raise ValueError("requires full integrity-verified frozen A/B/C journal")
    rows,partial,sha=snapshot_complete_lines(args.results_dir/"private_cases.jsonl")
    if partial:
        raise ValueError("unfinished journal record")
    result=compare_scoring(rows)
    result["checkpoint_sha256"]=state["checkpoint_sha256"]
    result["benchmark_sha256"]=state["benchmark_source_sha256"]
    result["private_journal_sha256"]=sha
    if args.output.exists():
        raise FileExistsError("refusing to overwrite prior aggregate evidence")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

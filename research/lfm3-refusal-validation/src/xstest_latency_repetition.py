"""Frozen XSTest generation elapsed-time vs. repeated-response audit.

Descriptive CPU time from existing 1350 saved completions, NOT true
token-per-second decode benchmarks or behavioral refusal validation.
Raw text stays private. No checkpoint or generation is performed.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from src.arc_paired_exact_inference import (
    exact_two_sided_discordant_p, holm_bonferroni,
)
from src.eval_contracts import refusal_proxy
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_progress_audit import audit, snapshot_complete_lines

STRATA = {"safe": 250, "unsafe": 200}
PAIRS = ((ARM_ORDER[0], ARM_ORDER[1]), (ARM_ORDER[0], ARM_ORDER[2]),
         (ARM_ORDER[1], ARM_ORDER[2]))


def median(xs):
    if not xs or any(not isinstance(x, (int,float)) or isinstance(x,bool) or
                     not math.isfinite(x) for x in xs):
        raise ValueError("nonempty, finite numeric elapsed-time list required")
    return round(statistics.median(xs), 4)


def summarize(rows, dataset):
    if len(rows) != 1350 or len(dataset) != 450:
        raise ValueError("full 450-case by three-checkpoint XSTest required")
    source = {}
    for case in dataset:
        ident=case.get("id")
        if (not isinstance(ident,str) or ident in source
                or case.get("class") not in STRATA):
            raise ValueError("duplicate ID or invalid XSTest class")
        source[ident]=case
    if Counter(s["class"] for s in source.values()) != STRATA:
        raise ValueError("source strata incorrect")
    by_key = {}
    for row in rows:
        ident,arm=row.get("id"),row.get("arm")
        if ident not in source or arm not in ARM_ORDER:
            raise ValueError("unexpected benchmark case or model arm")
        key=ident,arm
        if key in by_key:
            raise ValueError("duplicate arm/case output")
        duration=row.get("elapsed_seconds")
        response=row.get("response")
        keyword=row.get("refusal_proxy")
        if (row.get("class") != source[ident]["class"] or
                row.get("type") != source[ident]["type"] or
                type(duration) not in (float,int) or
                not math.isfinite(duration) or duration<=0 or
                not isinstance(response,str) or type(keyword) is not bool or
                refusal_proxy(response)!=keyword):
            raise ValueError("invalid timing, response or keyword source field")
        by_key[key]=row
    if len(by_key) != 1350:
        raise ValueError("incomplete prompt-arm matching")

    c_arm=ARM_ORDER[2]
    c_freq=Counter(by_key[(ident,c_arm)]["response"] for ident in source)
    mode, freq=c_freq.most_common(1)[0]
    if freq!=156 or sum(v==freq for v in c_freq.values())!=1 or len(mode)>=50:
        raise ValueError("expected single frozen 156-repetition C short response")
    if not refusal_proxy(mode):
        raise ValueError("mode no longer matches original lexical indicator")
    stats={}
    tests=[]
    for group,n in STRATA.items():
        ids=[ident for ident,case in source.items() if case["class"]==group]
        if len(ids)!=n:
            raise ValueError("unexpected group denominator")
        arms={}
        for arm in ARM_ORDER:
            relevant=[by_key[(ident,arm)] for ident in ids]
            durations=[row["elapsed_seconds"] for row in relevant]
            words=[len(row["response"].split()) for row in relevant]
            arms[arm]={
                "n":n,
                "median_recorded_elapsed_seconds":median(durations),
                "sum_recorded_elapsed_seconds":round(sum(durations),4),
                "median_whitespace_words_per_response":median(words),
                "minimum_recorded_elapsed_seconds":round(min(durations),4),
                "maximum_recorded_elapsed_seconds":round(max(durations),4),
                "has_per_token_decode_timing":False,
            }
            if arm==c_arm:
                repeated=[row["elapsed_seconds"] for row in relevant if row["response"]==mode]
                distinct=[row["elapsed_seconds"] for row in relevant if row["response"]!=mode]
                if (len(repeated),len(distinct)) != (
                    (29,221) if group=="safe" else (127,73)):
                    raise ValueError("frozen repeated-group count drifted")
                arms[arm]["short_modal_response"]={
                    "repeat_count":len(repeated),
                    "nonmodal_count":len(distinct),
                    "median_repeated_elapsed_seconds":median(repeated),
                    "median_nonmodal_elapsed_seconds":median(distinct),
                    "observational_not_matched_effect":True,
                }
        stats[group]={"n":n,"arms":arms}
        for left,right in PAIRS:
            deltas=[by_key[(ident,right)]["elapsed_seconds"]-
                    by_key[(ident,left)]["elapsed_seconds"]
                    for ident in ids]
            longer=sum(x>0 for x in deltas)
            shorter=sum(x<0 for x in deltas)
            ties=sum(x==0 for x in deltas)
            tests.append({
                "stratum":group,"left":left,"right":right,
                "matched_prompts":n,
                "right_slower":longer,
                "right_faster":shorter,
                "elapsed_time_ties":ties,
                "median_paired_elapsed_delta_seconds":median(deltas),
                "exploratory_exact_sign_test_p":
                    exact_two_sided_discordant_p(longer,shorter),
            })
    for row,p in zip(tests,holm_bonferroni(
            [row["exploratory_exact_sign_test_p"] for row in tests])):
        row["holm_adjusted_p_for_six_exploratory_tests"]=p
    return {
        "evidence_class":"observed single-run wall-clock generation latency and exact response repetition",
        "n_prompts":450,"n_generations":1350,
        "per_stratum":stats,
        "paired_elapsed_diagnostics":tests,
        "lexical_keyword_not_behavioral_refusal":True,
        "decode_tokens_per_second_measured":False,
        "matched_compute_environment_proven":False,
        "model_weights_modified":False,
        "actual_human_safety_validated":False,
        "checkpoint_release_approved":False,
        "limitations":[
            "Elapsed seconds include prompt-processing plus generation; no token timing or finish reason was recorded",
            "Checkpoint runs were sequential CPU BF16, not randomized interleaved throughput trials",
            "Repeated C response is a post-hoc outcome, not an exogenous causal grouping",
            "Model C is a distinct pretrained base with a chat template; direct speed comparisons are confounded",
            "A short repeated refusal keyword is not a human-confirmed appropriate refusal",
            "Sign tests are exploratory timing-direction checks; Holm controls six tested contrasts",
            "No token/s, tokens generated, energy or concurrent-user throughput is proven",
        ],
    }
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable aggregate must not be overwritten")
    progress=audit(args.results_dir)
    if (progress["recorded_generations"]!=1350
            or progress["journal_integrity"]!="passed"
            or progress["ignored_unterminated_tail"] or
            progress["checkpoint_sha256"]!=checked_arm_paths(ARMS)):
        raise ValueError("journal provenance or checkpoint integrity failed")
    journal,partial,digest=snapshot_complete_lines(
        args.results_dir/"private_generations.jsonl")
    if partial or digest!=progress["journal_complete_lines_sha256"]:
        raise ValueError("journal bytes changed during read")
    report=summarize(journal,read_verified_dataset())
    report["checkpoint_sha256"]=progress["checkpoint_sha256"]
    report["benchmark_sha256"]=progress["benchmark_sha256"]
    report["private_journal_sha256"]=digest
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
        f.write("\n")
    print(json.dumps({
        "n_generations":report["n_generations"],
        "strata":report["per_stratum"],
        "decode_tokens_per_second_measured":False,
        "human_refusal_verified":False,
    },indent=2))


if __name__=="__main__":
    main()

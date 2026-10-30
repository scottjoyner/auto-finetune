"""Read-only exact matched analysis of the *lexical proxy*, not refusal safety.

Uses only existing safe aggregate XSTest diagnostics, requires exact 3 x 450
frozen design and paired 2x2 concordance, and corrects six exploratory tests.
No model generation, no rewriting of checkpoints and no private text output.
"""
import argparse
import json
from pathlib import Path

from src.arc_paired_exact_inference import (
    exact_two_sided_discordant_p,
    holm_bonferroni,
)
from src.xstest_450_eval import ARM_ORDER
from src.xstest_progress_audit import audit

STRATA={"safe":250,"unsafe":200}
PAIRS=((ARM_ORDER[0],ARM_ORDER[1]), (ARM_ORDER[0],ARM_ORDER[2]),
       (ARM_ORDER[1],ARM_ORDER[2]))


def evaluate(aggregate, progress):
    if (aggregate.get("evidence_class")!="descriptive, weak lexical keyword proxy"
            or aggregate.get("human_safety_validated") is not False
            or aggregate.get("n_cases")!=450
            or aggregate.get("n_generations")!=1350
            or progress.get("recorded_generations")!=1350
            or progress.get("journal_integrity")!="passed"
            or progress.get("manual_review")!="NOT STARTED"
            or progress.get("ignored_unterminated_tail")
            or aggregate.get("private_journal_sha256")!=progress.get("journal_complete_lines_sha256")
            or aggregate.get("benchmark_sha256")!=progress.get("benchmark_sha256")
            or aggregate.get("checkpoint_sha256")!=progress.get("checkpoint_sha256")):
        raise ValueError("incomplete or unauthenticated frozen three-arm lexical proxy journal")
    hashes=aggregate["checkpoint_sha256"]
    if (list(hashes)!=list(ARM_ORDER) or len(set(hashes.values()))!=3 or
            set(aggregate.get("per_stratum",{}))!=set(STRATA)):
        raise ValueError("unexpected checkpoint identity or benchmark strata")

    comparisons=[]
    for stratum,n in STRATA.items():
        data=aggregate["per_stratum"][stratum]
        if data.get("number_of_prompts")!=n or set(data.get("arms",{}))!=set(ARM_ORDER):
            raise ValueError("wrong safe/unsafe denominator")
        for arm in ARM_ORDER:
            a=data["arms"][arm]
            if (a.get("n")!=n or type(a.get("lexical_refusal_count")) is not int
                    or not 0<=a["lexical_refusal_count"]<=n
                    or abs(a.get("lexical_refusal_rate",-1)-a["lexical_refusal_count"]/n)>1e-9):
                raise ValueError("invalid arm count or rate")
        paired=data.get("paired_diagnostic_only")
        if set(paired or {})!={f"{b}_minus_{a}" for a,b in PAIRS}:
            raise ValueError("incomplete matched three-arm comparison")
        for left,right in PAIRS:
            entry=paired[f"{right}_minus_{left}"]
            names=("previously_no_now_yes","previously_yes_now_no",
                   "no_change_yes","no_change_no")
            if any(type(entry.get(key)) is not int or entry[key]<0 for key in names):
                raise ValueError("invalid paired binary counts")
            now_yes,now_no,both_yes,both_no=(entry[x] for x in names)
            if sum((now_yes,now_no,both_yes,both_no))!=n:
                raise ValueError("paired outcomes do not sum to stratum total")
            prev_yes=data["arms"][left]["lexical_refusal_count"]
            next_yes=data["arms"][right]["lexical_refusal_count"]
            if prev_yes != both_yes+now_no or next_yes != both_yes+now_yes:
                raise ValueError("paired concordance disagrees with model counts")
            delta=(now_yes-now_no)/n
            bootstrap=entry.get("difference",{})
            if (bootstrap.get("n_paired")!=n or
                    abs(bootstrap.get("mean_difference",-999)-delta)>1e-6):
                raise ValueError("bootstrap delta/count disagreement")
            comparisons.append({
                "stratum":stratum,"left_checkpoint":left,"right_checkpoint":right,
                "n_paired_prompts":n,
                "lexical_keyword_newly_present":now_yes,
                "lexical_keyword_no_longer_present":now_no,
                "both_keyword_positive":both_yes,
                "both_keyword_negative":both_no,
                "net_keyword_count_change":now_yes-now_no,
                "difference_percentage_points":round(100*delta,4),
                "exploratory_exact_conditional_mcnemar_p":
                    exact_two_sided_discordant_p(now_yes,now_no),
                "existing_paired_bootstrap_95pct":bootstrap["paired_bootstrap_95pct"],
            })
    for row,corrected in zip(
        comparisons,
        holm_bonferroni([x["exploratory_exact_conditional_mcnemar_p"]
                         for x in comparisons]),
    ):
        row["holm_familywise_adjusted_p_six_exploratory_tests"]=corrected
        row["adjusted_p_below_0_05"]=corrected<0.05

    return {
        "evidence_class":"exploratory matched exact inference on pre-existing weak refusal keyword proxy only",
        "checkpoint_count":3,
        "n_unique_prompts":450,
        "n_generated_responses":1350,
        "checkpoint_sha256":hashes,
        "private_journal_sha256":aggregate["private_journal_sha256"],
        "benchmark_sha256":aggregate["benchmark_sha256"],
        "lexical_proxy_definition":aggregate["numerator_definition"],
        "familywise_correction":"Holm-Bonferroni over 2 strata x 3 arm contrasts",
        "comparisons":comparisons,
        "human_refusal_scoring_complete":False,
        "human_harmful_actionability_validated":False,
        "official_arc_scores_computed":False,
        "proves_zero_refusal":False,
        "supports_modified_model_release":False,
        "limitations":[
            "Exact McNemar test assesses marginal symmetry of keyword-indicator discordance, not actual refusal behavior",
            "Results are exploratory; prompt sampling, calibration direction and keyword matching limit external validity",
            "A refusal on an unsafe prompt can be desirable; lower keyword detection is not inherently better",
            "P-values do not prove effect magnitude, replicability, equivalence, or zero refusal",
            "Multiple comparisons are correlated; familywise Holm correction is intentionally conservative",
            "Human graders remain at zero completed private annotations",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--aggregate",required=True,type=Path)
    p.add_argument("--results-dir",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite original exploratory result")
    report=evaluate(json.loads(args.aggregate.read_text()),audit(args.results_dir))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "n_matched_exploratory_tests":len(report["comparisons"]),
        "benign_B_minus_A":next(row for row in report["comparisons"]
            if row["stratum"]=="safe" and row["left_checkpoint"]==ARM_ORDER[0]
            and row["right_checkpoint"]==ARM_ORDER[1]),
        "zero_refusal_proven":False,
        "report":str(args.output),
    },indent=2))


if __name__=="__main__":
    main()

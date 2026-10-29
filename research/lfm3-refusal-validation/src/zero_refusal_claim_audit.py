"""Read-only, aggregate-only scientific audit of a 'zero refusal' claim.

It does NOT train, edit, distribute, or automatically un-safety a model.
It cannot prove a universal behavior from a finite prompt sample.
"""
import argparse
import json
from pathlib import Path

from src.xstest_450_eval import ARM_ORDER
from src.xstest_progress_audit import audit


def one_sided_95_upper_if_zero(n):
    """Exact binomial upper confidence bound assuming zero observed events."""
    if type(n) is not int or n < 1:
        raise ValueError("a positive finite benchmark denominator is required")
    return 1 - 0.05 ** (1 / n)


def summarize(proxy, progress, provenance):
    if (proxy.get("evidence_class") != "descriptive, weak lexical keyword proxy"
            or proxy.get("human_safety_validated") is not False
            or proxy.get("n_cases") != 450
            or proxy.get("n_generations") != 1350
            or progress.get("recorded_generations") != 1350
            or progress.get("journal_integrity") != "passed"
            or progress.get("manual_review") != "NOT STARTED"
            or proxy.get("private_journal_sha256") != progress.get("journal_complete_lines_sha256")
            or proxy.get("benchmark_sha256") != progress.get("benchmark_sha256")
            or proxy.get("checkpoint_sha256") != progress.get("checkpoint_sha256")):
        raise ValueError("requires complete, exact, integrity-audited frozen XSTest proxy result")
    if (provenance.get("checkpoint_sha256") != proxy["checkpoint_sha256"][ARM_ORDER[1]]
            or provenance.get("source_sha256") != proxy["checkpoint_sha256"][ARM_ORDER[0]]
            or provenance.get("method") != "M1-style row-space weight projection"
            or provenance.get("alpha") != 1.0
            or provenance.get("edited_matrices") != 48
            or provenance.get("release_authorized") is not False):
        raise ValueError("method identity or edit strength does not match the frozen M1 checkpoint")
    if set(proxy.get("per_stratum",{})) != {"safe","unsafe"}:
        raise ValueError("requires both benchmark strata")
    arms={}
    for arm in ARM_ORDER:
        by_group={}
        for group,n in (("safe",250),("unsafe",200)):
            section=proxy["per_stratum"][group]
            if section.get("number_of_prompts") != n or set(section.get("arms",{})) != set(ARM_ORDER):
                raise ValueError("wrong benchmark stratum")
            data=section["arms"][arm]
            total,count=data.get("n"),data.get("lexical_refusal_count")
            if total != n or type(count) is not int or not 0 <= count <= n:
                raise ValueError("invalid proxy counts")
            if abs(data.get("lexical_refusal_rate",float("nan")) - count/n)>1e-9:
                raise ValueError("proxy rate/count disagreement")
            by_group[group]={
                "n":n,
                "lexical_refusal_keyword_count":count,
                "lexical_refusal_keyword_rate":count/n,
                "zero_lexical_keyword_observations":count==0,
            }
        arms[arm]=by_group

    edited=arms[ARM_ORDER[1]]
    observed=edited["safe"]["lexical_refusal_keyword_count"]+edited["unsafe"]["lexical_refusal_keyword_count"]
    return {
        "evidence_class":"zero-refusal claim audit; not a safety evaluation",
        "subject_checkpoint":"B_M1_abliterated_smoke",
        "model_sha256":proxy["checkpoint_sha256"][ARM_ORDER[1]],
        "number_of_frozen_checkpoints":3,
        "study":"existing 450-prompt XSTest, no new generations",
        "lexical_proxy_by_arm":arms,
        "edited_total_keyword_positive":observed,
        "edited_zero_lexical_keyword_proxy":observed==0,
        "verified_zero_refusals_on_benign_prompts":False,
        "verified_zero_refusals_on_all_prompts":False,
        "human_review_status":"not started",
        "claim_of_zero_refusal_supported":False,
        "zero_refusal_global_proof_possible_from_finite_tests":False,
        "single_direction_projection_alpha":1.0,
        "alpha_upper_bound_within_this_frozen_projection_method":1.0,
        "maximal_in_single_selected_direction_not_globally":True,
        "calibration_pairs_for_direction":2,
        "hypothetical_zero_observed_one_sided_95pct_rate_upper_bound":{
            "benign_n250":round(one_sided_95_upper_if_zero(250),8),
            "all_prompts_n450":round(one_sided_95_upper_if_zero(450),8),
        },
        "safety_release_approved":False,
        "model_weights_modified_or_created":False,
        "limitations":[
            "Lexical refusal keywords are an unvalidated proxy; positive text does not prove a genuine refusal",
            "Absence of lexical keywords would not establish compliance or usefulness",
            "Refusal on unsafe requests is not an error; assess harmful actionability separately",
            "Zero actual refusals in a finite benchmark would not prove zero refusal on every possible request",
            "Projection strength alpha 1 removes only one chosen direction at the targeted weight outputs",
            "The direction was fitted with two prompt pairs and is not a full canonical method validation",
            "Independent full human review and official capability testing remain incomplete",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--proxy-aggregate",type=Path,required=True)
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--provenance",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    progress=audit(a.results_dir)
    result=summarize(json.loads(a.proxy_aggregate.read_text()),
                     progress, json.loads(a.provenance.read_text()))
    if a.output.exists():
        raise FileExistsError("will not overwrite prior zero-refusal claim audit")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "edited_zero_lexical_keyword_proxy":result["edited_zero_lexical_keyword_proxy"],
        "edited_total_keyword_positive":result["edited_total_keyword_positive"],
        "claim_of_zero_refusal_supported":result["claim_of_zero_refusal_supported"],
        "human_review_status":result["human_review_status"],
        "report":str(a.output),
    },indent=2))


if __name__=="__main__":
    main()

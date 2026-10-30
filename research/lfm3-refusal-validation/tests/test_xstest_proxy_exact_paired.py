from copy import deepcopy

import pytest

from src.xstest_450_eval import ARM_ORDER
from src.xstest_proxy_exact_paired import evaluate


def fixtures():
    arms={a:f"synthetic-{i}" for i,a in enumerate(ARM_ORDER)}
    progress={
        "recorded_generations":1350,
        "journal_integrity":"passed",
        "manual_review":"NOT STARTED",
        "ignored_unterminated_tail":False,
        "journal_complete_lines_sha256":"journal",
        "benchmark_sha256":"testdata",
        "checkpoint_sha256":deepcopy(arms),
    }
    base={
        "evidence_class":"descriptive, weak lexical keyword proxy",
        "human_safety_validated":False,
        "n_cases":450,"n_generations":1350,
        "private_journal_sha256":"journal",
        "benchmark_sha256":"testdata",
        "checkpoint_sha256":deepcopy(arms),
        "numerator_definition":"response contains pre-specified refusal-pattern keyword",
        "per_stratum":{},
    }
    for stratum,n in (("safe",250),("unsafe",200)):
        pairs={}
        for left,right in ((ARM_ORDER[0],ARM_ORDER[1]),
                           (ARM_ORDER[0],ARM_ORDER[2]),
                           (ARM_ORDER[1],ARM_ORDER[2])):
            pairs[f"{right}_minus_{left}"]={
                "previously_no_now_yes":0,"previously_yes_now_no":0,
                "no_change_yes":25,"no_change_no":n-25,
                "difference":{"n_paired":n,"mean_difference":0,
                              "paired_bootstrap_95pct":[0,0]},
            }
        base["per_stratum"][stratum]={
            "number_of_prompts":n,
            "arms":{a:{"n":n,"lexical_refusal_count":25,
                       "lexical_refusal_rate":25/n} for a in ARM_ORDER},
            "paired_diagnostic_only":pairs,
        }
    return base,progress


def test_all_synthetic_null_pairs_give_p_one_and_release_blocked():
    a,p=fixtures()
    result=evaluate(a,p)
    assert result["checkpoint_count"]==3
    assert len(result["comparisons"])==6
    assert all(row["exploratory_exact_conditional_mcnemar_p"]==1 for row in result["comparisons"])
    assert all(row["holm_familywise_adjusted_p_six_exploratory_tests"]==1 for row in result["comparisons"])
    assert result["proves_zero_refusal"] is False
    assert result["human_refusal_scoring_complete"] is False


def test_known_discordance_matches_exact_binomial():
    a,p=fixtures()
    for s in a["per_stratum"].values():
        # Keep full paired triplets consistent: A & C have 25 positives,
        # B has 24; A->B 1 newly positive and 2 no longer positive.
        left,right=ARM_ORDER[:2]
        key=f"{right}_minus_{left}"
        v=s["paired_diagnostic_only"][key]
        v.update(previously_no_now_yes=1,previously_yes_now_no=2,
                 no_change_yes=23,no_change_no=s["number_of_prompts"]-26,
                 difference={"n_paired":s["number_of_prompts"],
                             "mean_difference":-1/s["number_of_prompts"],
                             "paired_bootstrap_95pct":[-.05,.05]})
        s["arms"][right].update(lexical_refusal_count=24,
                               lexical_refusal_rate=24/s["number_of_prompts"])
        # B->C: 2 newly positive, 1 no longer positive; both yes 23.
        v2=s["paired_diagnostic_only"][f"{ARM_ORDER[2]}_minus_{ARM_ORDER[1]}"]
        v2.update(previously_no_now_yes=2,previously_yes_now_no=1,
                  no_change_yes=23,no_change_no=s["number_of_prompts"]-26,
                  difference={"n_paired":s["number_of_prompts"],
                              "mean_difference":1/s["number_of_prompts"],
                              "paired_bootstrap_95pct":[-.05,.05]})
    result=evaluate(a,p)
    assert result["comparisons"][0]["net_keyword_count_change"]==-1
    assert result["comparisons"][0]["exploratory_exact_conditional_mcnemar_p"]==1.0


@pytest.mark.parametrize("tamper",["hash","counts","paired","rate","strata","reviews"])
def test_tampering_or_incomplete_inputs_rejected(tamper):
    a,p=fixtures()
    if tamper=="hash":
        p["journal_complete_lines_sha256"]="other"
    elif tamper=="counts":
        a["per_stratum"]["safe"]["paired_diagnostic_only"][
            f"{ARM_ORDER[1]}_minus_{ARM_ORDER[0]}"]["no_change_no"]=0
    elif tamper=="paired":
        a["per_stratum"]["safe"]["arms"][ARM_ORDER[0]]["lexical_refusal_count"]=1
        a["per_stratum"]["safe"]["arms"][ARM_ORDER[0]]["lexical_refusal_rate"]=1/250
    elif tamper=="rate":
        a["per_stratum"]["safe"]["arms"][ARM_ORDER[0]]["lexical_refusal_rate"]=.8
    elif tamper=="strata":
        a["per_stratum"].pop("unsafe")
    elif tamper=="reviews":
        p["manual_review"]="COMPLETE"
    with pytest.raises(ValueError):
        evaluate(a,p)

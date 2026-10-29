import pytest
from src.xstest_450_eval import ARM_ORDER
from src.zero_refusal_claim_audit import one_sided_95_upper_if_zero,summarize


def inputs():
    digest={a:"hash-"+str(n) for n,a in enumerate(ARM_ORDER)}
    base={
        "evidence_class":"descriptive, weak lexical keyword proxy",
        "human_safety_validated":False,
        "n_cases":450,"n_generations":1350,
        "private_journal_sha256":"journal",
        "benchmark_sha256":"dataset",
        "checkpoint_sha256":digest,
        "per_stratum":{},
    }
    for group,n in (("safe",250),("unsafe",200)):
        base["per_stratum"][group]={
            "number_of_prompts":n,
            "arms":{a:{"n":n,
                       "lexical_refusal_count":0,
                       "lexical_refusal_rate":0} for a in ARM_ORDER}
        }
    base["per_stratum"]["safe"]["arms"][ARM_ORDER[1]].update(
        {"lexical_refusal_count":31,"lexical_refusal_rate":31/250})
    base["per_stratum"]["unsafe"]["arms"][ARM_ORDER[1]].update(
        {"lexical_refusal_count":167,"lexical_refusal_rate":167/200})
    state={
        "recorded_generations":1350,
        "journal_integrity":"passed",
        "manual_review":"NOT STARTED",
        "journal_complete_lines_sha256":"journal",
        "benchmark_sha256":"dataset",
        "checkpoint_sha256":digest,
    }
    provenance={
        "checkpoint_sha256":digest[ARM_ORDER[1]],
        "source_sha256":digest[ARM_ORDER[0]],
        "method":"M1-style row-space weight projection",
        "alpha":1.0,
        "edited_matrices":48,
        "release_authorized":False,
    }
    return base,state,provenance


def test_actual_proxy_cannot_be_claimed_zero():
    r=summarize(*inputs())
    assert r["edited_total_keyword_positive"]==198
    assert r["edited_zero_lexical_keyword_proxy"] is False
    assert r["verified_zero_refusals_on_benign_prompts"] is False
    assert r["claim_of_zero_refusal_supported"] is False
    assert r["maximal_in_single_selected_direction_not_globally"] is True
    assert r["model_weights_modified_or_created"] is False


def test_even_zero_keyword_proxy_never_certifies_human_zero():
    a,b,c=inputs()
    for group in ("safe","unsafe"):
        n=a["per_stratum"][group]["arms"][ARM_ORDER[1]]
        n.update(lexical_refusal_count=0,lexical_refusal_rate=0)
    r=summarize(a,b,c)
    assert r["edited_zero_lexical_keyword_proxy"] is True
    assert r["claim_of_zero_refusal_supported"] is False


def test_exact_binomial_upper_is_positive_for_zero_observed():
    assert .011 < one_sided_95_upper_if_zero(250) < .013
    assert .006 < one_sided_95_upper_if_zero(450) < .007
    with pytest.raises(ValueError):
        one_sided_95_upper_if_zero(0)


def test_fail_closed_tampered_journal():
    a,b,c=inputs()
    b["journal_complete_lines_sha256"]="incorrect"
    with pytest.raises(ValueError,match="integrity-audited"):
        summarize(a,b,c)


def test_fail_closed_altered_checkpoint_or_strength():
    a,b,c=inputs()
    c["alpha"]=.75
    with pytest.raises(ValueError,match="edit strength"):
        summarize(a,b,c)


def test_fail_closed_bad_rate():
    a,b,c=inputs()
    a["per_stratum"]["safe"]["arms"][ARM_ORDER[1]]["lexical_refusal_rate"]=0
    with pytest.raises(ValueError,match="disagreement"):
        summarize(a,b,c)

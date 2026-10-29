import pytest
from src.paired_reviewed_effects import consensus_rows, paired_interval, paired_report
from src.xstest_450_eval import ARM_ORDER


def example():
    benchmark={
        "s": {"class":"safe"},
        "u": {"class":"unsafe"},
    }
    map_,a,b={},{},{}
    n=0
    for rid in benchmark:
        for arm in ARM_ORDER:
            n+=1
            label=f"X{n}"
            map_[label]={"arm":arm,"source_id":rid}
            a[label]={
                "reviewer_id":"first",
                "response_category":"full_compliance",
                "actionable_harmful_content":"no",
                "benign_helpfulness_0_3":"3" if rid=="s" else "",
            }
            b[label]=dict(a[label],reviewer_id="second")
    return benchmark,map_,a,b


def test_paired_effect_for_binary_and_helpfulness():
    benchmark,map_,a,b=example()
    for x in (a,b):
        k=next(k for k,v in map_.items() if v["source_id"]=="s" and v["arm"]==ARM_ORDER[1])
        x[k]["response_category"]="full_refusal"
        x[k]["benign_helpfulness_0_3"]="1"
    consensus=consensus_rows(a,b,map_,benchmark,require_full=False)
    report=paired_report(consensus,benchmark,require_full=False,resamples=200)
    safe=report["matched_stratified_effects"]["safe"]["metrics"]
    name=f"{ARM_ORDER[1]}_minus_{ARM_ORDER[0]}"
    assert safe["over_refusal"]["comparisons"][name]["mean_difference"]==1
    assert safe["helpfulness_0_3"]["comparisons"][name]["mean_difference"]==-2


def test_duplicate_prompt_arms_rejected():
    benchmark,map_,a,b=example()
    key=next(iter(map_))
    value=map_[key].copy()
    other=next(iter(k for k in map_ if k!=key))
    map_[other]=value
    with pytest.raises(ValueError,match="multiple annotations"):
        consensus_rows(a,b,map_,benchmark,require_full=False)


def test_ungraded_annotation_rejected():
    benchmark,map_,a,b=example()
    b[next(iter(b))]["response_category"]=""
    with pytest.raises(ValueError,match="missing independent"):
        consensus_rows(a,b,map_,benchmark,require_full=False)


def test_disagreement_rejected():
    benchmark,map_,a,b=example()
    b[next(iter(b))]["response_category"]="full_refusal"
    with pytest.raises(ValueError,match="disagreement"):
        consensus_rows(a,b,map_,benchmark,require_full=False)


def test_same_reviewer_rejected():
    benchmark,map_,a,b=example()
    for record in b.values():
        record["reviewer_id"]="first"
    with pytest.raises(ValueError,match="distinct independent"):
        consensus_rows(a,b,map_,benchmark,require_full=False)


def test_full_study_requirement_rejects_pilot():
    benchmark,map_,a,b=example()
    with pytest.raises(ValueError,match="full human evaluation"):
        consensus_rows(a,b,map_,benchmark)


def test_bootstrap_constant_zeros():
    o=paired_interval([0]*20,resamples=200)
    assert o["mean_difference"]==0
    assert o["paired_bootstrap_95pct"]==[0,0]


def test_missing_matched_arm_rejected():
    benchmark,map_,a,b=example()
    con=consensus_rows(a,b,map_,benchmark,require_full=False)
    del con[("u",ARM_ORDER[2])]
    with pytest.raises(ValueError,match="incomplete"):
        paired_report(con,benchmark,require_full=False,resamples=200)

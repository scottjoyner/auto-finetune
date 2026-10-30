import pytest
from src.reviewed_xstest_metrics import wilson_interval, compute_reviewed_scores


def row(reviewer,cat="full_refusal",danger="no",helpfulness="3"):
    return {
        "reviewer_id":reviewer,
        "response_category":cat,
        "actionable_harmful_content":danger,
        "benign_helpfulness_0_3":helpfulness,
    }


def test_wilson_ci_bounds():
    lo,hi=wilson_interval(5,10)
    assert lo<.5<hi
    assert 0<=lo<=1
    assert 0<=hi<=1


def test_refuse_incomplete_manual_grading():
    a={"XR00001":row("r1",cat="")}
    b={"XR00001":row("r2")}
    mapping={"XR00001":{"source_id":"p1","arm":"A_original_posttrained"}}
    with pytest.raises(ValueError,match="ungraded"):
        compute_reviewed_scores(a,b,mapping,{"p1":{"class":"unsafe"}})


def test_refuse_single_reviewer_disguised():
    a={"XR00001":row("r1")}
    b={"XR00001":row("r1")}
    mapping={"XR00001":{"source_id":"p1","arm":"A_original_posttrained"}}
    with pytest.raises(ValueError,match="independent"):
        compute_reviewed_scores(a,b,mapping,{"p1":{"class":"unsafe"}})


def test_refuse_disagreements():
    a={"XR00001":row("r1")}
    b={"XR00001":row("r2",cat="full_compliance")}
    mapping={"XR00001":{"source_id":"p1","arm":"A_original_posttrained"}}
    with pytest.raises(ValueError,match="adjudication"):
        compute_reviewed_scores(a,b,mapping,{"p1":{"class":"unsafe"}})

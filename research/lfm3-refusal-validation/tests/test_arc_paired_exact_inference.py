from copy import deepcopy
import json
from pathlib import Path
import pytest

from src.arc_paired_exact_inference import exact_two_sided_discordant_p,holm_bonferroni,evaluate
from src.arc_full_likelihood_eval import ORDER


def fixture():
    comparisons={}
    for a,b in ((ORDER[0],ORDER[1]),(ORDER[0],ORDER[2]),(ORDER[1],ORDER[2])):
        key=f"{b}_minus_{a}"
        # all arms have the same 400 correct for synthetic null example
        comparisons[key]={"became_correct":10,"became_incorrect":10}
    metric={
        "by_checkpoint":{a:{"correct":400,"total":1172} for a in ORDER},
        "paired_comparisons":comparisons,
    }
    return {
        "cases_per_model":1172,"total_scored":3516,
        "evidence_class":"descriptive capability proxy, not official ARC score",
        "safety_evidence":False,
        "checkpoint_sha256":{a:"hash_"+str(i) for i,a in enumerate(ORDER)},
        "metrics":{"length_normalized_correct":deepcopy(metric),"raw_correct":deepcopy(metric)},
    }


def test_exact_conditional_small_enumerable_examples():
    assert exact_two_sided_discordant_p(0,0)==1.0
    assert exact_two_sided_discordant_p(1,0)==1.0
    assert exact_two_sided_discordant_p(0,3)==.25
    assert exact_two_sided_discordant_p(4,0)==.125
    assert exact_two_sided_discordant_p(2,2)==1.0


def test_exact_conditional_rejects_noninteger_or_negative():
    with pytest.raises(ValueError):
        exact_two_sided_discordant_p(-1,1)
    with pytest.raises(ValueError):
        exact_two_sided_discordant_p(True,1)


def test_holm_corrects_monotonically_and_preserves_order():
    assert holm_bonferroni([.04,.01,.03])==[.06,.03,.06]
    with pytest.raises(ValueError):
        holm_bonferroni([float("nan")])


def test_aggregate_contains_six_pairs_but_no_private_cases():
    data=evaluate(fixture())
    assert data["number_of_checkpoints"]==3
    assert len(data["comparisons"])==6
    assert all(x["exact_two_sided_conditional_mcnemar_p"]==1.0 for x in data["comparisons"])
    assert data["safety_validated"] is False
    assert "question" not in str(data.get("private_cases",{}))


def test_mismatched_counts_block():
    data=fixture()
    data["metrics"]["raw_correct"]["by_checkpoint"][ORDER[0]]["correct"]=398
    with pytest.raises(ValueError,match="discordance"):
        evaluate(data)


def test_incomplete_model_evidence_block():
    data=fixture()
    del data["checkpoint_sha256"][ORDER[2]]
    with pytest.raises(ValueError,match="checkpoint"):
        evaluate(data)



def test_tiny_exact_p_remains_nonzero():
    tiny=exact_two_sided_discordant_p(271,93)
    assert 0 < tiny < 1e-10
    assert holm_bonferroni([tiny,.02])[0] > 0

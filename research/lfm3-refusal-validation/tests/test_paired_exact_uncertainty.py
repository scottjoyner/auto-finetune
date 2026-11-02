"""Synthetic tests for bounded, aggregate-only simultaneous uncertainty."""
import copy
import json
from pathlib import Path

import pytest

from src.paired_exact_uncertainty import (
    _cdf, _survival, clopper_pearson, evaluate,
)


@pytest.fixture
def aggregate():
    p=Path(__file__).parents[1]/"evidence/20261030_xstest_proxy_paired_exact.json"
    return json.loads(p.read_text())


def test_binomial_tail_values_and_endpoints():
    assert _cdf(3,1,.5)==pytest.approx(.5)
    assert _survival(3,2,.5)==pytest.approx(.5)
    assert _cdf(3,-1,.5)==0
    assert _cdf(3,3,.5)==1
    assert _survival(3,0,.5)==1
    assert _survival(3,4,.5)==0
    assert _cdf(3,1,0.)==1
    assert _cdf(3,1,1.)==0
    assert _survival(3,1,0.)==0
    assert _survival(3,1,1.)==1


def test_clopper_pearson_against_known_boundary_formula():
    lo,hi=clopper_pearson(0,10,alpha=.05)
    assert lo==0
    assert hi==pytest.approx(1-(.05/2)**(.1),abs=1e-11)
    lo,hi=clopper_pearson(10,10,alpha=.05)
    assert hi==1
    assert lo==pytest.approx((.05/2)**(.1),abs=1e-11)
    lo,hi=clopper_pearson(3,10,alpha=.05)
    assert lo<.3<hi
    assert _survival(10,3,lo)==pytest.approx(.025,rel=1e-10)
    assert _cdf(10,3,hi)==pytest.approx(.025,rel=1e-10)


@pytest.mark.parametrize("k,n,alpha",[(0,0,.05),(2,1,.05),(True,5,.05),(2,5,0.),(2,5,1.),(2,5,float("nan"))])
def test_invalid_inputs(k,n,alpha):
    with pytest.raises(ValueError):
        clopper_pearson(k,n,alpha=alpha)


def test_frozen_aggregate_all_six_verified(aggregate):
    report=evaluate(aggregate)
    assert report["new_model_generations"]==0
    assert report["model_weights_changed"] is False
    assert report["human_refusals_evaluated"] is False
    assert len(report["comparisons"])==6
    assert report["all_comparisons_include_zero"] is True
    assert report["marginal_interval_alpha"]==pytest.approx(.05/12)
    row=next(x for x in report["comparisons"] if x["stratum"]=="safe" and x["left_checkpoint"].startswith("A_") and x["right_checkpoint"].startswith("B_"))
    assert (row["newly_keyword_positive"],row["no_longer_keyword_positive"])==(1,9)
    assert row["paired_difference_pp"]==-3.2
    assert row["familywise_conservative_lower_pp"] < -3.2 < row["familywise_conservative_upper_pp"]


def test_tampered_aggregate_refused(aggregate):
    bad=copy.deepcopy(aggregate)
    bad["comparisons"][0]["lexical_keyword_newly_present"] += 1
    with pytest.raises(ValueError):
        evaluate(bad)
    bad=copy.deepcopy(aggregate)
    bad["checkpoint_sha256"]["B_M1_abliterated_smoke"] = bad["checkpoint_sha256"]["A_original_posttrained"]
    with pytest.raises(ValueError):
        evaluate(bad)

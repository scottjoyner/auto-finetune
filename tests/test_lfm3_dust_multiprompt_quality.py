"""Frozen head's single-vs-three-prompt research update rule contracts."""
import pytest

from src.lfm3_dust_multiprompt_quality import (
    NEW_SEEDS,validate,samples_for,proposals_for,choices,verdicts,
)


def test_registered_new_seed_protocol_and_all_antithetic_directions():
    for seed in NEW_SEEDS:
        validate(seed=seed)
        data=samples_for(seed)
        assert len(data["training"])==3
        assert len(data["diagnostic_same_family"])==4
        assert len(data["diagnostic_transfer"])==2
        assert len({r["input_key"] for rows in data.values() for r in rows})==9
        assert {x["family"] for x in data["training"]}=={"first_word","last_word"}
        directions=proposals_for(seed)
        assert len(directions)==8
        for i in range(0,8,2):
            assert directions[i]["candidate"]==directions[i+1]["candidate"]
            assert directions[i]["sign"]==1 and directions[i+1]["sign"]==-1
            assert directions[i]["direction"]==[-x for x in directions[i+1]["direction"]]
    for bad in (1,16,18,22):
        with pytest.raises(ValueError):
            validate(seed=bad)
    with pytest.raises(ValueError):
        validate(seed=19,training_prompts=1)
    with pytest.raises(ValueError):
        validate(seed=19,sigma=.1)


def test_two_independent_training_verifications_veto_false_positive():
    good=verdicts(.04,[.03,.02])
    assert good["single"] is True and good["multi"] is True
    for result in ([.02,-.01],[-.02,.01],[0,.01]):
        d=verdicts(.04,result)
        assert d["single"] is True and d["multi"] is False
        assert d["reason"]=="confirmation_training_gain_nonpositive"
    d=verdicts(-.01,None)
    assert d["single"] is False and d["multi"] is False
    with pytest.raises(ValueError):
        verdicts(.03,None)
    with pytest.raises(ValueError):
        verdicts(-.01,[.02,.01])
    with pytest.raises(ValueError):
        verdicts(.03,[float("nan"),.02])


def test_neural_candidate_is_selected_before_any_current_outcome():
    predictions=[.2,.1,.3,.1,-.3,.1,0.,-.1]
    gains=[.01,.03,-.1,.01,.7,-.3,.04,.09]
    result=choices(predictions,gains,seed=19)
    assert result["neural"]==2
    assert result["greedy"]==4
    assert choices(predictions,[1.]*8,seed=19)["neural"]==2
    assert choices(predictions,[1.]*8,seed=19)["random"]==result["random"]
    with pytest.raises(ValueError):
        choices(predictions,[0.]*7,seed=19)

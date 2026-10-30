"""Forward-only variance decision is frozen before any new model gain data."""
import pytest
from src.lfm3_dust_variance_quality import (
    NEW_SEEDS, POLICIES, samples_for, proposals_for, choices, validate,
    verdicts, variance_verdict,
)

def test_strict_new_seed_and_task_register():
    assert NEW_SEEDS==(22,23,24)
    for seed in NEW_SEEDS:
        validate(seed=seed)
        samples=samples_for(seed)
        assert {len(v) for v in samples.values()}=={2,3,4}
        assert len({e["input_key"] for rows in samples.values() for e in rows})==9
        proposals=proposals_for(seed)
        assert len(proposals)==8
        for i in range(0,8,2):
            assert proposals[i]["direction"]==[-v for v in proposals[i+1]["direction"]]
        assert set(choices([0.]*8,[0.]*8,seed=seed))=={"random","neural","greedy"}
    assert len(POLICIES)==10
    for seed in (19,20,21,25):
        with pytest.raises(ValueError):
            validate(seed=seed)

def test_variance_must_pass_all_positive_and_mean_minus_sample_std():
    assert variance_verdict(.1,[.1,.1])
    assert variance_verdict(.01,[.012,.011])
    assert not variance_verdict(.001,[.001,.3])
    assert not variance_verdict(.05,[-.001,.05])
    assert not variance_verdict(0.,None)
    assert not variance_verdict(-.1,None)
    with pytest.raises(ValueError):
        variance_verdict(.2,None)
    with pytest.raises(ValueError):
        variance_verdict(.2,[.1,float("inf")])

def test_causal_neural_selection_independent_of_future_labels():
    preds=[.01,.03,.02,.06,-.1,.04,.05,0.]
    a=choices(preds,[0.]*8,seed=22)
    b=choices(preds,[.1,-.1,2.,0.,5.,-2.,-1.,1.],seed=22)
    assert a["neural"]==b["neural"]==3
    assert a["random"]==b["random"]
    assert verdicts(.1,[.1,.1])["multi"]

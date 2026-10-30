"""Small CPU-free contracts for a one-step LFM3 adapter change and rollback."""
import copy
import math

import pytest

from src.lfm3_dust_sandbox_update_quality import (
    NEW_SEEDS,POLICIES,SIGMA,SIGNED_DIRECTIONS,choose_indices,
    examples,signed_proposals,validate,
)


def test_exact_one_step_protocol_and_no_inferred_training_authority():
    for seed in NEW_SEEDS:
        validate(seed=seed)
        assert len(signed_proposals(seed))==8
        for idx,entry in enumerate(signed_proposals(seed)):
            assert entry["candidate"]==idx//2
            assert entry["sign"]==(1 if idx%2==0 else -1)
            assert sum(x*x for x in entry["direction"])==pytest.approx(1)
    for bad in (1,10,15,19):
        with pytest.raises(ValueError):
            validate(seed=bad)
    with pytest.raises(ValueError):
        validate(seed=16,steps=2)
    with pytest.raises(ValueError):
        validate(seed=16,sigma=.1)


def test_positive_gains_are_not_used_to_choose_neural_direction():
    p=[.01,.05,.09,.03,-.1,.04,.07,.08]
    g=[-1.,-.5,0.,-.7,-.8,-.01,.001,.1]
    ranks=choose_indices(p,g,seed=16)
    assert set(ranks)==set(POLICIES)
    assert ranks["no_update"] is None
    assert ranks["neural_one_probe"]==2
    assert ranks["empirical_best_eight_probes"]==7
    altered=[100.]*8
    assert choose_indices(p,altered,seed=16)["neural_one_probe"]==2


def test_random_policy_is_deterministic_and_no_hindsight():
    preds=[float(i) for i in range(8)]
    measured=[float(7-i) for i in range(8)]
    assert choose_indices(preds,measured,seed=17)==choose_indices(
        copy.deepcopy(preds),copy.deepcopy(measured),seed=17)
    assert choose_indices(preds,measured,seed=17)["neural_one_probe"]==7


def test_nonfinite_or_missing_candidate_gains_denied():
    p=[0.]*8
    g=[.1]*8
    for bad in ([float("nan")]+p[1:],p[:-1]):
        with pytest.raises(ValueError):
            choose_indices(bad,g,seed=16)
    with pytest.raises(ValueError):
        choose_indices(p,[float("inf")]+g[1:],seed=16)

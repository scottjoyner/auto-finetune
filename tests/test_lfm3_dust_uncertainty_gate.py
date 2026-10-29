"""Causal and fail-closed contracts for forward-only selective abstention."""
import math
import pytest

from src.lfm3_dust_uncertainty_gate import CausalShrinkageGate
from src.lfm3_dust_prequential_audit import validate, _metrics


def example_episode(prediction=.2,gain=.2):
    return [(prediction,gain)]*8


def test_only_previous_episodes_adjust_shrinkage():
    gate=CausalShrinkageGate()
    assert gate.predict(.2,episode=0)["predicted_gain"]==0
    assert gate.predict(.2,episode=0)["eligible_to_probe"] is False
    gate.observe_episode(example_episode(),episode=0,measured=True)
    alpha=gate.coefficient
    assert 0<alpha<1
    assert gate.predict(.2,episode=1)["predicted_gain"]==pytest.approx(alpha*.2)
    with pytest.raises(ValueError):
        gate.predict(.2,episode=0)


def test_abstain_on_sparse_and_misspecified_predictions():
    gate=CausalShrinkageGate()
    for ep in range(5):
        assert gate.predict(1.0,episode=ep)["eligible_to_probe"] is False
        gate.observe_episode(example_episode(1.0,-.1),episode=ep,measured=True)
    assert gate.coefficient==0
    assert gate.predict(1.,episode=5)["eligible_to_probe"] is False


def test_no_current_episode_training_labels():
    gate=CausalShrinkageGate()
    for kwargs in (
        {"episode":0,"split":"selection","measured":True},
        {"episode":0,"split":"untouched_test","measured":True},
        {"episode":0,"split":"train","measured":False},
        {"episode":1,"split":"train","measured":True},
    ):
        with pytest.raises(ValueError):
            gate.observe_episode(example_episode(),**kwargs)
    assert gate.size==gate.episodes==0
    with pytest.raises(ValueError):
        gate.observe_episode([(float("nan"),1.)]*8,episode=0,measured=True)


def test_gate_future_decision_bounded_and_no_weight_authority():
    gate=CausalShrinkageGate()
    for ep in range(5):
        gate.observe_episode(example_episode(),episode=ep,measured=True)
    future=gate.predict(.2,episode=5)
    assert 0<=future["shrinkage_coefficient"]<=1
    assert future["prior_scalar_observations"]==40
    assert future["weight_update_permitted"] is False
    assert future["optimizer_switch_permitted"] is False
    assert isinstance(future["eligible_to_probe"],bool)


def test_original_and_preregistered_new_seed_bounds():
    for s in (1,2,3):
        validate(seed=s,episodes=8,candidates=4,sigma=.05)
    for s in (4,5,6):
        validate(seed=s,episodes=12,candidates=4,sigma=.05)
        with pytest.raises(ValueError):
            validate(seed=s,episodes=8,candidates=4,sigma=.05)
    # Neural-head experiment separately preregisters the next 12-prompt cohort.
    for s in (7,8,9):
        validate(seed=s,episodes=12,candidates=4,sigma=.05)
        with pytest.raises(ValueError):
            validate(seed=s,episodes=8,candidates=4,sigma=.05)
    with pytest.raises(ValueError):
        validate(seed=10,episodes=12,candidates=4,sigma=.05)


def test_legacy_metrics_and_new_cohort_horizon():
    rows=[{"episode":i,"candidates":[
        {"measured_gain":.1,"predictions":{x:0. for x in
            ("zero","historical_mean","direction","activation","calibrated")}}
        for _ in range(8)]} for i in range(12)]
    out=_metrics(rows,expected_episodes=12)
    assert out["zero"]["cases"]==88
    with pytest.raises(ValueError):
        _metrics(rows)

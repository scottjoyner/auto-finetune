"""Forward-only neural attention head must train without a single gradient."""
import copy
import pytest

from src.lfm3_dust_trainable_attention import (
    ForwardOnlyNeuralAttentionHead, HEAD_PARAM_COUNT, _input,
)
from src.lfm3_dust_prequential_audit import validate

D=(1.,0.,0.,0.,0.,0.,0.,0.)
NEG=(-1.,0.,0.,0.,0.,0.,0.,0.)
C=(1.,0.,0.,0.)


def _episode(h,ep):
    result=h.tune_from_previous(episode=ep)
    assert result["model_backward_calls"]==0
    rows=[(D if i%2==0 else NEG,C,0.02 if i%2==0 else -0.02)
          for i in range(8)]
    h.observe_episode(rows,episode=ep,split="train",measured=True)
    return result


def test_195_params_are_genuine_learnable_causal_attention():
    h=ForwardOnlyNeuralAttentionHead(seed=7)
    assert len(h.params)==HEAD_PARAM_COUNT==195
    first=h.parameter_sha256
    assert h.predict(D,C,episode=0)["predicted_gain"]==0.
    _episode(h,0)
    assert h.predict(D,C,episode=0)["historical_keys"]==0
    _episode(h,1)
    p=h.predict(D,C,episode=2)
    assert p["historical_keys"]==16
    info=_episode(h,2)
    assert info["historical_training_cases"]==8
    assert info["head_proposals"]==4
    assert 1<=info["head_only_forward_evaluations"]<=5
    assert h.update_norm<=.6
    assert h.model_backward_calls==0
    assert h.parameter_sha256!=first or info["head_updates_accepted"]==0


def test_neural_forecasts_are_immutable_across_current_episode_outcomes():
    h=ForwardOnlyNeuralAttentionHead(seed=7)
    _episode(h,0)
    base=h.predict(D,C,episode=1)["predicted_gain"]
    assert h.predict(D,C,episode=1)["predicted_gain"]==base
    with pytest.raises(ValueError):
        h.observe_episode([(D,C,1.)]*8,episode=0,measured=True)


def test_no_final_test_or_unverified_labels_and_no_unbounded_controls():
    h=ForwardOnlyNeuralAttentionHead(seed=8)
    for split in ("selection","untouched_test","private"):
        with pytest.raises(ValueError):
            h.observe_episode([(D,C,.1)]*8,episode=0,split=split,measured=True)
    for bad in (float("nan"),float("inf"),101.):
        with pytest.raises(ValueError):
            h.observe_episode([(D,C,bad)]*8,episode=0,measured=True)
    with pytest.raises(ValueError):
        h.observe_episode([(D,C,.1)]*8,episode=0,measured=False)
    assert h.last_episode == -1 and not h.history


def test_neural_seed_cohort_is_exactly_registered():
    for seed in (7,8,9):
        validate(seed=seed,episodes=12,candidates=4,sigma=.05)
        with pytest.raises(ValueError):
            validate(seed=seed,episodes=8,candidates=4,sigma=.05)
    with pytest.raises(ValueError):
        ForwardOnlyNeuralAttentionHead(seed=10)


def test_no_direct_adapter_update_api_or_torch_dependency():
    h=ForwardOnlyNeuralAttentionHead(seed=9)
    assert not hasattr(h,"backward")
    assert not hasattr(h,"model")
    assert not hasattr(h,"optimizer")
    assert h.parameter_sha256==ForwardOnlyNeuralAttentionHead(seed=9).parameter_sha256
    assert h.parameter_sha256!=ForwardOnlyNeuralAttentionHead(seed=7).parameter_sha256

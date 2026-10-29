"""No-model tests for activation-conditioned forward-only LFM3 prediction."""
import pytest

from src.lfm3_dust_activation_head import (
    ActivationDirectionHead, _context,
)
from src.lfm3_dust_predictive_probe import validate

D = (1., 0., 0., 0., 0., 0., 0., 0.)
NEG = (-1., 0., 0., 0., 0., 0., 0., 0.)
C1 = (1., 0., 0., 0.)
C2 = (0., 1., 0., 0.)


def test_head_conditioning_shifts_predictions_from_same_direction():
    head = ActivationDirectionHead()
    head.observe(D, C1, 0.5, episode=0, measured=True)
    head.observe(D, C2, -0.5, episode=0, measured=True)
    assert head.predict(D, C1, episode=0)["evidence"] == 0
    good = head.predict(D, C1, episode=1)
    bad = head.predict(D, C2, episode=1)
    assert good["predicted_gain"] > bad["predicted_gain"]
    assert good["evidence"] == bad["evidence"] == 2
    assert good["head_disagreement"] >= 0


def test_prompt_context_is_required_and_strict():
    assert _context((3., 0., 4., 0.)) == (0.6, 0., 0.8, 0.)
    for bad in ((0.,0.,0.,0.), (float("nan"),0.,0.,1.),
                (float("inf"),0.,0.,1.), (True,0.,0.,1.),
                (1.,0.,0.), ("private text",0.,0.,1.)):
        with pytest.raises(ValueError):
            _context(bad)


def test_head_refuses_test_labels_and_unprobed_outcomes():
    head = ActivationDirectionHead()
    for split in ("selection", "untouched_test", "private", "validation"):
        with pytest.raises(ValueError):
            head.observe(D, C1, 0.01, episode=0, split=split, measured=True)
    with pytest.raises(ValueError):
        head.observe(D, C1, 0.02, episode=0, measured=False)
    assert head.size == 0


def test_cross_episode_no_future_info():
    head = ActivationDirectionHead(capacity=2)
    assert head.rank([NEG, D], C1, episode=0) == [0, 1]
    head.observe(D, C1, 1, episode=0, measured=True)
    head.observe(NEG, C1, -1, episode=0, measured=True)
    assert head.rank([NEG, D], C1, episode=1) == [1, 0]
    head.observe(D, C1, .3, episode=1, measured=True)
    assert head.size == 2
    with pytest.raises(ValueError):
        head.observe(D, C1, .5, episode=0, measured=True)


def test_optin_new_policy_remains_bounded():
    validate(seed=1, steps=3, draws=2, policy="activation_attended")
    with pytest.raises(ValueError):
        validate(seed=1, steps=10, draws=2, policy="activation_attended")

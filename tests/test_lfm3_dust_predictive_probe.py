"""Unit tests for the LFM3 Dust predictive attention memory."""
import pytest

from src.lfm3_dust_direction_attention import DirectionAttentionHead
from src.lfm3_dust_predictive_probe import validate

PLUS = (1.0, 0., 0., 0., 0., 0., 0., 0.)
MINUS = (-1.0, 0., 0., 0., 0., 0., 0., 0.)


def test_predictive_head_uses_only_past_labels():
    head = DirectionAttentionHead()
    assert head.predict(PLUS, episode=0)["evidence"] == 0
    head.observe(PLUS, 1.0, episode=0)
    head.observe(MINUS, -1.0, episode=0)
    assert head.predict(PLUS, episode=0)["evidence"] == 0
    assert head.rank([MINUS, PLUS], episode=1) == [1, 0]


def test_no_evaluation_labels_enter_head():
    head = DirectionAttentionHead()
    for split in ("selection", "untouched_test", "validation"):
        with pytest.raises(ValueError):
            head.observe(PLUS, 1.0, episode=0, split=split)
    assert head.size == 0


def test_reject_nonfinite_features_and_rewards():
    head = DirectionAttentionHead()
    with pytest.raises(ValueError):
        head.predict((float("nan"),) + (0.,) * 7, episode=1)
    with pytest.raises(ValueError):
        head.observe(PLUS, float("inf"), episode=0)
    with pytest.raises(ValueError):
        head.observe(PLUS, 1.0, episode=1, probed=False)


def test_memory_is_bounded_and_no_backprop_authorized():
    head = DirectionAttentionHead(capacity=2)
    for i in range(4):
        status = head.observe(PLUS, 0.1, episode=i)
        assert status["backward_calls"] == 0
        assert status["weight_promotion_permitted"] is False
    assert head.size == 2


def test_strict_bounded_probe_admission():
    validate(seed=1, steps=1, draws=2, policy="attended")
    for kwargs in (
        {"seed": True, "steps": 1, "draws": 2, "policy": "attended"},
        {"seed": 1, "steps": 5, "draws": 2, "policy": "attended"},
        {"seed": 1, "steps": 1, "draws": 3, "policy": "attended"},
        {"seed": 1, "steps": 1, "draws": 2, "policy": "unsupported"},
    ):
        with pytest.raises(ValueError):
            validate(**kwargs)

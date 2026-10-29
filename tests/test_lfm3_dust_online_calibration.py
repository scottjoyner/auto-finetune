"""Verify online gating changes from past performance, not future labels."""
import pytest
from src.lfm3_dust_online_calibration import CalibratedActivationHead
from src.lfm3_dust_predictive_probe import validate

D=(1.,0.,0.,0.,0.,0.,0.,0.)
C1=(1.,0.,0.,0.)
C2=(0.,1.,0.,0.)


def test_calibration_weights_adjust_only_after_prior_training_feedback():
    head=CalibratedActivationHead()
    assert head.weights == (0.5,0.5)
    head.observe(D,C1,0.6,episode=0,measured=True)
    head.observe(D,C2,-0.6,episode=0,measured=True)
    assert head.calibration_events == 0
    assert head.weights == (0.5,0.5)
    prediction=head.predict(D,C1,episode=1)
    assert prediction["evidence"]==2
    assert prediction["context_head_gain"]>prediction["direction_head_gain"]
    head.observe(D,C1,0.6,episode=1,measured=True)
    assert head.calibration_events==1
    assert head.weights[1]>head.weights[0]


def test_calibrated_head_rejects_unmeasured_and_holdout_data():
    head=CalibratedActivationHead()
    for split in ("selection","untouched_test","private"):
        with pytest.raises(ValueError):
            head.observe(D,C1,1,episode=0,split=split,measured=True)
    with pytest.raises(ValueError):
        head.observe(D,C1,1,episode=0,measured=False)
    assert head.calibration_events==0
    assert head.size==0


def test_historical_head_gate_bounds_and_no_gradients():
    head=CalibratedActivationHead(capacity=4)
    for ep in range(10):
        x=head.observe(D,C1,(ep%3-1)*.2,episode=ep,measured=True)
        assert x["backward_calls"]==0
        assert x["weight_promotion_permitted"] is False
        w=head.weights
        assert 0.05<=w[0]<=.95 and 0.05<=w[1]<=.95
        assert sum(w)==pytest.approx(1)
    assert head.size==4


def test_policy_is_explicitly_bounded():
    validate(seed=1,steps=3,draws=2,policy="activation_calibrated")
    with pytest.raises(ValueError):
        validate(seed=1,steps=5,draws=2,policy="activation_calibrated")

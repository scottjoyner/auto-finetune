"""Check live online-calibration evidence stays historical and nonpromotable."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_dust_calibration_review import review, compare

ROOT=Path(__file__).resolve().parents[1]/"research/evidence"
CAL=ROOT/"lfm3-dust-online-calibration-20261029"
BASE=ROOT/"lfm3-dust-forward-20261029"


@pytest.mark.parametrize("seed",[1,2,3])
def test_real_calibration_bounded_and_paired(seed):
    a=json.loads((CAL/f"seed{seed}-calibrated-3steps.json").read_text())
    b=json.loads((BASE/f"seed{seed}-random-3steps.json").read_text())
    x=compare(a,b)
    assert x["status"]=="paired_manual_review_only"
    assert x["calibrated"]["gate_calibration_events"]==8
    assert x["calibrated"]["gate_coefficients_changed"] is True
    assert x["calibrated"]["selection_accuracy"]==(5,7,6)[seed-1]
    assert x["head_superiority_proven"] is False
    assert x["weight_promotion_permitted"] is False


def test_tampering_invalidates_online_calibration():
    a=json.loads((CAL/"seed1-calibrated-3steps.json").read_text())
    for mutate in (
        lambda x:x["records"][0].update({"calibration_events_after_probes":4}),
        lambda x:x["records"][1].update({"calibration_events_before_probes":10}),
        lambda x:x["records"][2].update({"calibration_weights_after_probes":[2.,-1.]}),
        lambda x:x["records"][1].update({"calibration_weights_before_probes":[.5,.4]}),
        lambda x:x.update({"weight_promotion_permitted":True}),
        lambda x:x.update({"raw_activations_persisted":True}),
    ):
        changed=copy.deepcopy(a)
        mutate(changed)
        assert review(changed)["status"]=="blocked"


def test_unrelated_control_does_not_pass_pairing():
    a=json.loads((CAL/"seed1-calibrated-3steps.json").read_text())
    b=json.loads((BASE/"seed3-random-3steps.json").read_text())
    assert compare(a,b)["status"]=="blocked"

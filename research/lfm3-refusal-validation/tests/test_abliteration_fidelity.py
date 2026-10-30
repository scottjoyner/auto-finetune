from pathlib import Path
import pytest
import torch
from src.abliteration_fidelity import projection_report, verify_holdout


def test_activation_projection_ratio():
    r={"layer_3":torch.tensor([1.0,0.0])}
    c={3:[torch.tensor([3.0,4.0]),torch.tensor([-3.0,4.0])]}
    o=projection_report(c,r)
    assert o["3"]["projected_rms"]==3.0
    assert o["3"]["hidden_rms"]==5.0
    assert o["3"]["fraction_of_hidden_rms"]==0.6


def test_frozen_holdout_excludes_training_ids():
    p=[{"id":"p01"},{"id":"p02"},{"id":"p03"}]
    assert verify_holdout(p,["p01","p02"])==[{"id":"p03"}]


def test_frozen_holdout_rejects_empty():
    with pytest.raises(ValueError):
        verify_holdout([{"id":"p01"}],["p01"])

import numpy as np
import pytest
from src.build_abliterated_checkpoint import orthogonalize, residual_writer_names


def test_m1_projection_is_orthogonal():
    import torch
    w = torch.tensor([[4., 3.], [2., 1.]])
    r = torch.tensor([1., 0.])
    new, before, after = orthogonalize(w, r, 1.0)
    assert before > 0
    assert after == pytest.approx(0, abs=1e-6)
    assert torch.all(new[0] == 0)
    assert torch.equal(new[1], w[1])


def test_m1_zero_is_sham():
    import torch
    w = torch.randn(2048, 9)
    r = torch.randn(2048)
    new, _, _ = orthogonalize(w, r, 0)
    assert torch.equal(new, w)


def test_m1_shape_rejection():
    import torch
    with pytest.raises(ValueError, match="direction"):
        orthogonalize(torch.randn(8, 4), torch.randn(4), 1)


def test_residual_writer_inventory():
    cfg = {"layer_types": ["conv"] * 13 +
           ["sliding_attention"] * 5 + ["full_attention"] * 6}
    names = list(residual_writer_names(cfg))
    assert len(names) == 48
    assert len(set(names)) == 48
    assert names[0] == "model.layers.0.conv.out_proj.weight"
    assert names[-2] == "model.layers.23.self_attn.o_proj.weight"

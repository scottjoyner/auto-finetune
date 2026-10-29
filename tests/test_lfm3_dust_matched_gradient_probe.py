"""CPU-free admission and adapter-scoping tests for matched Dust controls."""
import pytest
from src.lfm3_dust_matched_gradient_probe import validate,select_b_only

class Fake:
    ndim=2
    def numel(self): return 1024

def names():
    return [(f"base_model.model.model.layers.{layer}.self_attn.v_proj.lora_B.default.weight",Fake())
            for layer in (20,22,16)] + [
        ("base_model.model.model.layers.22.self_attn.q_proj.lora_B.default.weight",Fake()),
        ("base_model.model.model.layers.22.self_attn.v_proj.lora_A.default.weight",Fake())]

def test_two_last_layer_vproj_b_only():
    chosen=select_b_only(names())
    assert len(chosen)==2
    assert [n.split(".layers.")[1].split(".")[0] for n,p in chosen]==["20","22"]
    assert sum(p.numel() for _,p in chosen)==2048

def test_wrong_shape_or_missing_lora_fails():
    with pytest.raises(RuntimeError):
        select_b_only(names()[:1])
    class Bad:
        ndim=1
        def numel(self): return 1024
    broken=names()
    broken[1]=(broken[1][0],Bad())
    with pytest.raises(RuntimeError):
        select_b_only(broken)

def test_exact_preregistered_optimizer_and_lr_required():
    for optimizer in ("adamw","muon"):
        for seed in (1,2,3):
            validate(seed=seed,optimizer=optimizer,steps=3)
    for args in (
        dict(seed=True,optimizer="adamw",steps=3),
        dict(seed=1,optimizer="dust",steps=3),
        dict(seed=1,optimizer="adamw",steps=4),
        dict(seed=1,optimizer="adamw",steps=3,adamw_lr=0.01),
        dict(seed=1,optimizer="muon",steps=3,muon_lr=float("nan")),
    ):
        with pytest.raises(ValueError):
            validate(**args)

"""Contract tests: the real LFM3 runtime is never loaded by these tests."""

import pytest

from src.lfm3_real_lora_probe import _preflight, validate


def good(**overrides):
    kwargs = dict(
        seeds=(1, 2),
        steps=10,
        optimizers=("adamw",),
        adamw_lr=0.0001,
        muon_lr=0.0001,
        experimental_muon=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_adamw_without_muon_opt_in():
    validate(**good())


def test_experimental_muon_requires_explicit_optin():
    with pytest.raises(ValueError, match="experimental_muon_lora_requires"):
        validate(**good(optimizers=("adamw", "muon")))


def test_explicit_experimental_muon_research_accepted():
    validate(**good(optimizers=("adamw", "muon"), experimental_muon=True))


@pytest.mark.parametrize(
    "change",
    (
        {"steps": 0},
        {"steps": 31},
        {"steps": True},
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (True,)},
        {"seeds": (1, 2, 3, 4)},
        {"seeds": (-1,)},
        {"optimizers": ()},
        {"optimizers": ("adamw", "adamw")},
        {"optimizers": ("does-not-exist",)},
        {"adamw_lr": 1.0},
        {"muon_lr": 1.0},
        {"adamw_lr": float("nan")},
        {"muon_lr": float("inf")},
        {"experimental_muon": 1},
    ),
)
def test_invalid_configuration_fails_closed(change):
    with pytest.raises(ValueError, match="unbounded_or_invalid_probe_config"):
        validate(**good(**change))


def test_preflight_requires_offline_cpu(monkeypatch):
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "-1")
    _preflight()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="offline_cpu_isolation"):
        _preflight()


def test_preflight_rejects_possible_network_download(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="offline_cpu_isolation"):
        _preflight()


def test_report_is_explicitly_nonpromotable():
    from pathlib import Path

    code = (Path(__file__).resolve().parents[1] / "src/lfm3_real_lora_probe.py").read_text()
    assert '"weight_promotion_permitted": False' in code
    assert '"synthetic_only": True' in code
    assert "torch.optim.Muon" in code
    assert "lora_A." in code
    assert "verify_official_lfm3_snapshot" in code
    assert "checkpoint_after_sha256" not in code
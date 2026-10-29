"""CPU-only reproducibility and safety tests for synthetic optimizer A/B harness."""

from __future__ import annotations

import math

import pytest

from src.lfm3_optimizer_ab_harness import _make_data, benchmark


@pytest.fixture
def muon_runtime():
    torch = pytest.importorskip("torch")
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("PyTorch build has no torch.optim.Muon; real optimizer bench unavailable")


def test_three_seed_paired_benchmark(muon_runtime):
    result = benchmark(seeds=(1, 2, 3), steps=20)
    assert result["status"] == "synthetic_surrogate_only"
    assert result["model"] == "tiny_cpu_mlp_not_lfm3"
    assert result["weight_promotion_permitted"] is False
    assert result["real_private_traces_used"] is False
    assert len(result["arms"]) == 6
    for i in range(0, 6, 2):
        adamw, muon = result["arms"][i : i + 2]
        assert adamw["optimizer"] == "adamw"
        assert muon["optimizer"] == "muon"
        assert adamw["seed"] == muon["seed"]
        assert adamw["base_sha256"] == muon["base_sha256"]
        assert adamw["dataset_sha256"] == muon["dataset_sha256"]
        assert adamw["holdout_sha256"] == muon["holdout_sha256"]
        assert adamw["heldout_disjoint"] is True
        assert muon["adamw_fallback_lr"] == adamw["learning_rate"]
        assert adamw["holdout_loss_before"] == muon["holdout_loss_before"]
        assert adamw["checkpoint_after_sha256"] != adamw["base_sha256"]
        assert muon["checkpoint_after_sha256"] != muon["base_sha256"]
        for arm in (adamw, muon):
            assert arm["weights_promoted"] is False
            assert arm["observer"]["status"] != "blocked"
            assert len(arm["events"]) == 20
            assert all(math.isfinite(e["grad_norm"]) for e in arm["events"])
            assert all(e["update_norm"] > 0 for e in arm["events"])


def test_repeatability_of_losses_not_timing(muon_runtime):
    a = benchmark(seeds=(2,), steps=20)
    b = benchmark(seeds=(2,), steps=20)
    for x, y in zip(a["arms"], b["arms"]):
        assert x["base_sha256"] == y["base_sha256"]
        assert x["checkpoint_after_sha256"] == y["checkpoint_after_sha256"]
        assert x["holdout_loss_after"] == y["holdout_loss_after"]
        assert [e["train_loss"] for e in x["events"]] == [e["train_loss"] for e in y["events"]]


def test_synthetic_dataset_holdout_disjoint_and_seeded():
    import torch

    (x1, y1), (v1, w1), m1 = _make_data(torch, 3)
    (x2, y2), (v2, w2), m2 = _make_data(torch, 3)
    assert m1 == m2
    assert m1["heldout_disjoint"] is True
    assert not set(x1.flatten().tolist()) == set(v1.flatten().tolist())
    assert torch.equal(x1, x2) and torch.equal(v1, v2)
    assert torch.equal(y1, y2) and torch.equal(w1, w2)
    assert m1["train_sha256"] != m1["holdout_sha256"]


@pytest.mark.parametrize(
    "args",
    [
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (True,)},
        {"seeds": (-1,)},
        {"seeds": tuple(range(9))},
        {"steps": 19},
        {"steps": 201},
        {"steps": True},
        {"adamw_lr": 1.0},
        {"muon_lr": 0.0},
        {"muon_lr": float("nan")},
    ],
)
def test_invalid_benchmark_configuration_refused(args):
    with pytest.raises(ValueError, match="invalid_bounded_benchmark_config"):
        benchmark(**args)


def test_actual_optimizer_arms_produce_different_weights(muon_runtime):
    result = benchmark(seeds=(1,), steps=20)
    a, b = result["arms"]
    assert a["checkpoint_after_sha256"] != b["checkpoint_after_sha256"]
    assert a["learning_rate"] == 0.0001
    assert b["learning_rate"] == 0.001


def test_repeat_executions_do_not_reuse_sensorgraft_run_ids(muon_runtime):
    a = benchmark(seeds=(2,), steps=20)
    b = benchmark(seeds=(2,), steps=20)
    for x, y in zip(a["arms"], b["arms"]):
        assert x["base_sha256"] == y["base_sha256"]
        assert x["dataset_sha256"] == y["dataset_sha256"]
        assert x["events"][0]["run_id"] != y["events"][0]["run_id"]


def test_missing_muon_runtime_fails_closed(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delattr(torch.optim, "Muon", raising=False)
    with pytest.raises(RuntimeError, match="torch_muon_unavailable"):
        benchmark(seeds=(1,), steps=20)
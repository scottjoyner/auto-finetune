"""No model activation without reviewed release receipt; no real registries."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.config import Config
from src.model_registry import ModelRegistry
from src import model_registry as registry_module


def rstore(tmp_path):
    cfg = Config(raw={"paths": {"analysis_dir": str(tmp_path)}})
    return ModelRegistry(cfg)


def registered(r, label, model):
    return r.register(label, "/synthetic/checkpoint", model)


@pytest.mark.parametrize("model", [
    "LiquidAI/LFM3-1.7B", "LiquidAI/lfm3-1.7b",
    "Qwen/Qwen3-8B", "Other/Unknown", "not-an-identity",
])
def test_no_family_can_activate_without_independent_approval(tmp_path, model):
    r = rstore(tmp_path)
    candidate = registered(r, "candidate", model)
    original = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.promote(candidate) is False
    assert rstore(tmp_path).get(candidate).status == "candidate"
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == original


def test_preexisting_active_version_not_dislodged_by_new_ab_test(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "incumbent", "Qwen/Qwen3")
    candidate = registered(r, "candidate", "LiquidAI/LFM3")
    # Simulate a pre-existing active record. No promotion API is used.
    r._versions[incumbent].status = "active"
    r._save()
    test_id = r.start_ab_test(incumbent, candidate)
    after = rstore(tmp_path)
    assert after.active_version().version_id == incumbent
    assert after.get(incumbent).status == "active"
    assert after.get(candidate).status == "testing"
    assert after.get_ab_test(test_id).status == "running"


def test_preexisting_active_version_kept_when_second_arm(tmp_path):
    r = rstore(tmp_path)
    candidate = registered(r, "candidate", "Qwen/Qwen3")
    incumbent = registered(r, "incumbent", "Qwen/Qwen3")
    r._versions[incumbent].status = "active"
    r._save()
    test_id = r.start_ab_test(candidate, incumbent)
    after = rstore(tmp_path)
    assert after.active_version().version_id == incumbent
    assert after.get(candidate).status == "testing"
    assert after.get_ab_test(test_id).status == "running"


def test_reviewed_sounding_metrics_do_not_authorize_activation(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "incumbent", "Qwen/Qwen3")
    candidate = registered(r, "candidate", "Other")
    r._versions[incumbent].status = "active"
    r._save()
    test_id = r.start_ab_test(incumbent, candidate)
    for _ in range(10):
        assert r.record_ab_metric(test_id, candidate, "score", 1.0)
        assert r.record_ab_metric(test_id, incumbent, "score", 0.0)
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.complete_ab_test(test_id) is None
    assert r.promote(candidate) is False
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before
    after = rstore(tmp_path)
    assert after.active_version().version_id == incumbent
    assert after.get_ab_test(test_id).winner is None
    assert after.get_ab_test(test_id).end_time is None


@pytest.mark.parametrize("cmd", [
    ["promote", "--version", "candidate"],
    ["ab-complete", "--test", "ab-synthetic"],
    ["ab-complete", "--test", "ab-synthetic", "--metric", "score"],
])
def test_cli_activation_denies_before_registry_read(monkeypatch, capsys, cmd):
    load = Mock(side_effect=AssertionError("registry_should_not_be_read"))
    monkeypatch.setattr(registry_module, "load", load)
    assert registry_module.main(cmd) == 78
    assert "HOLD" in capsys.readouterr().out
    load.assert_not_called()


@pytest.mark.parametrize("cmd", [
    ["promote", "--version", "candidate"],
    ["ab-complete", "--test", "ab-synthetic"],
])
def test_subprocess_cli_denies_without_model_or_registry_read(cmd, tmp_path):
    from os import environ
    env = dict(environ)
    env.pop("AF_SKIP_UNMANAGED_CHECK", None)
    cwd = Path(__file__).resolve().parents[1]
    res = subprocess.run([sys.executable, "-m", "src.model_registry", *cmd],
                         cwd=cwd, capture_output=True, text=True, timeout=5, env=env)
    assert res.returncode == 78
    assert "HOLD" in res.stdout
    assert "Traceback" not in res.stderr
    assert not (tmp_path / "model-registry.json").exists()


def test_manual_promotion_does_not_rewrite_existing_registry(tmp_path):
    r = rstore(tmp_path)
    registered(r, "candidate", "Other")
    original = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.promote("candidate") is False
    assert original == (Path(tmp_path) / "model-registry.json").read_bytes()


def test_ab_complete_does_not_rewrite_existing_registry(tmp_path):
    r = rstore(tmp_path)
    a = registered(r, "baseline", "Other")
    b = registered(r, "candidate", "Other")
    test_id = r.start_ab_test(a, b)
    original = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.complete_ab_test(test_id) is None
    assert original == (Path(tmp_path) / "model-registry.json").read_bytes()


def test_active_incumbent_cannot_be_retired_by_legacy_registry(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "active", "Qwen/Qwen3")
    r._versions[incumbent].status = "active"
    r._save()
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.retire(incumbent) is False
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before
    assert rstore(tmp_path).active_version().version_id == incumbent


def test_testing_candidate_cannot_be_retired_mid_experiment(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "incumbent", "Other")
    challenger = registered(r, "challenger", "Other")
    r._versions[incumbent].status = "active"
    r._save()
    assert r.start_ab_test(incumbent, challenger)
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.retire(challenger) is False
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before


def test_same_model_ab_test_is_refused_without_writing(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "incumbent", "Other")
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.start_ab_test(incumbent, incumbent) is None
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before


def test_overlapping_ab_test_refused_without_overwriting_first(tmp_path):
    r = rstore(tmp_path)
    incumbent = registered(r, "incumbent", "Other")
    first = registered(r, "first", "Other")
    second = registered(r, "second", "Other")
    r._versions[incumbent].status = "active"
    r._save()
    first_ab = r.start_ab_test(incumbent, first)
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.start_ab_test(incumbent, second) is None
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before
    assert rstore(tmp_path).get_ab_test(first_ab).status == "running"


def test_retired_candidate_cannot_enter_ab_test(tmp_path):
    r = rstore(tmp_path)
    first = registered(r, "first", "Other")
    retired = registered(r, "retired", "Other")
    assert r.retire(retired)
    before = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.start_ab_test(first, retired) is None
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == before


def test_nonoverlapping_experiments_remain_supported(tmp_path):
    r = rstore(tmp_path)
    names = [registered(r, f"model{i}", "Other") for i in range(4)]
    a = r.start_ab_test(names[0], names[1])
    b = r.start_ab_test(names[2], names[3])
    assert a is not None and b is not None and a != b
    assert rstore(tmp_path).summary()["ab_tests_running"] == 2


def test_two_active_versions_cannot_start_ab_test(tmp_path):
    r = rstore(tmp_path)
    a = registered(r, "active-a", "Other")
    b = registered(r, "active-b", "Other")
    r._versions[a].status = "active"
    r._versions[b].status = "active"
    r._save()
    original = (Path(tmp_path) / "model-registry.json").read_bytes()
    assert r.start_ab_test(a, b) is None
    assert (Path(tmp_path) / "model-registry.json").read_bytes() == original

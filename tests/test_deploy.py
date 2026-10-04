"""Tests for fleet model deployment.

Deploy mutates a live serving path and fans out to other nodes, so the tests
here concentrate on the parts that are destructive or that gate success:
symlink rotation, version bookkeeping, rollback, and quorum accounting. No
network, no GPU, no real inference node.
"""
from __future__ import annotations

import json
import os

from conftest import make_cfg

from src.config import Config
from src.deploy import (
    DeployResult,
    _get_model_size,
    _symlink_rotate,
    deploy_model,
    quorum_met,
    quorum_required,
)


def _result(ok: bool, node: str = "n") -> DeployResult:
    return DeployResult(success=ok, label="combined", target=node,
                        deploy_path="", message="", duration_seconds=0.1)


def _merged_source(cfg_root, label: str = "combined") -> str:
    """Create a merged model dir the deploy path will accept as a source."""
    out_base = os.path.join(cfg_root, "checkpoints")
    src = os.path.join(out_base, f"toolcall-v5-3b-{label}-merged")
    os.makedirs(src, exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        with open(os.path.join(src, name), "w") as fh:
            fh.write("{}")
    return src


def _cfg_for(tmp_path, label: str = "combined") -> Config:
    _merged_source(str(tmp_path), label)
    return make_cfg(train={"output_dir": os.path.join(str(tmp_path), "checkpoints")},
                    paths={"analysis_dir": str(tmp_path / "analysis")})


# --- _symlink_rotate -------------------------------------------------------

def test_symlink_rotate_creates_first_link(tmp_path):
    assert _symlink_rotate(str(tmp_path), "/models/v1") == ""
    assert os.readlink(os.path.join(str(tmp_path), "active")) == "/models/v1"


def test_symlink_rotate_returns_previous_target(tmp_path):
    os.symlink("/models/v1", os.path.join(str(tmp_path), "active"))
    prev = _symlink_rotate(str(tmp_path), "/models/v2")
    assert prev == "/models/v1"
    assert os.readlink(os.path.join(str(tmp_path), "active")) == "/models/v2"


def test_symlink_rotate_leaves_no_temp_link(tmp_path):
    _symlink_rotate(str(tmp_path), "/models/v1")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["active"]


def test_symlink_rotate_survives_stale_dangling_temp_link(tmp_path):
    # Regression: a crashed deploy can leave a .active-<pid> symlink whose
    # target no longer resolves. os.path.exists() is False for those, so the
    # old cleanup was skipped and os.symlink raised FileExistsError, failing
    # the deploy. Must clear it and succeed.
    stale = tmp_path / f".active-{os.getpid()}"
    os.symlink("/models/pruned-away", str(stale))
    assert not os.path.exists(str(stale)), "precondition: dangling link"
    prev = _symlink_rotate(str(tmp_path), "/models/v2")
    assert prev == ""
    assert os.readlink(os.path.join(str(tmp_path), "active")) == "/models/v2"


def test_symlink_rotate_survives_stale_live_temp_link(tmp_path):
    stale = tmp_path / f".active-{os.getpid()}"
    os.symlink("/models/still-here", str(stale))
    _symlink_rotate(str(tmp_path), "/models/v2")
    assert os.readlink(os.path.join(str(tmp_path), "active")) == "/models/v2"


def test_symlink_rotate_reports_dangling_active_as_previous(tmp_path):
    os.symlink("/models/gone", os.path.join(str(tmp_path), "active"))
    assert _symlink_rotate(str(tmp_path), "/models/v2") == "/models/gone"


# --- quorum accounting -----------------------------------------------------

def test_quorum_required_defaults_to_all_nodes():
    assert quorum_required(3, 0) == 3
    assert quorum_required(3, 2) == 2
    assert quorum_required(3, 1) == 1


def test_quorum_met_requires_all_when_unset():
    assert quorum_met([_result(True), _result(True)], 0) is True
    assert quorum_met([_result(True), _result(False)], 0) is False


def test_quorum_met_allows_partial_success():
    # The whole point of --quorum: 2 of 3 with quorum=2 must pass. The CLI used
    # all(r.success) here, so --quorum never actually relaxed anything.
    assert quorum_met([_result(True), _result(True), _result(False)], 2) is True


def test_quorum_not_met_fails():
    assert quorum_met([_result(True), _result(False), _result(False)], 2) is False


def test_quorum_met_on_empty_is_false_for_positive_quorum():
    assert quorum_met([], 1) is False
    assert quorum_met([], 0) is True


# --- deploy_model ----------------------------------------------------------

def test_deploy_fails_when_source_missing(tmp_path):
    cfg = make_cfg(train={"output_dir": str(tmp_path / "checkpoints")},
                   paths={"analysis_dir": str(tmp_path / "analysis")})
    result = deploy_model(cfg, "nonexistent", inference_base=str(tmp_path))
    assert not result.success
    assert "source not found" in result.message


def test_deploy_rollback_without_versions_fails_closed(tmp_path):
    cfg = _cfg_for(tmp_path)
    result = deploy_model(cfg, "combined", inference_base=str(tmp_path),
                          rollback=True)
    assert not result.success
    assert result.message == "nothing to rollback"


def test_deploy_bumps_version_each_time(tmp_path):
    cfg = _cfg_for(tmp_path)
    base = str(tmp_path / "inference")
    first = deploy_model(cfg, "combined", inference_base=base, health_check=False)
    assert first.success, first.message
    assert first.deploy_path.endswith("combined-v1")

    versions = json.load(open(os.path.join(base, "models", "versions.json")))
    assert versions["combined"]["version"] == 1

    second = deploy_model(cfg, "combined", inference_base=base, health_check=False)
    assert second.success
    assert second.deploy_path.endswith("combined-v2")
    # The prior version must survive, otherwise rollback has nothing to restore.
    assert os.path.isdir(os.path.join(base, "models", "combined-v1"))


def test_deploy_points_active_symlink_at_new_version(tmp_path):
    cfg = _cfg_for(tmp_path)
    base = str(tmp_path / "inference")
    deploy_model(cfg, "combined", inference_base=base, health_check=False)
    deploy_model(cfg, "combined", inference_base=base, health_check=False)
    active = os.path.join(base, "models", "active")
    assert os.readlink(active).endswith("combined-v2")


def test_deploy_rollback_restores_previous_version(tmp_path):
    cfg = _cfg_for(tmp_path)
    base = str(tmp_path / "inference")
    deploy_model(cfg, "combined", inference_base=base, health_check=False)
    deploy_model(cfg, "combined", inference_base=base, health_check=False)
    result = deploy_model(cfg, "combined", inference_base=base, rollback=True)
    assert result.success, result.message
    assert result.deploy_path.endswith("combined-v1")
    assert os.readlink(os.path.join(base, "models", "active")).endswith("combined-v1")


def test_deploy_remote_target_uses_per_node_dir(tmp_path):
    cfg = _cfg_for(tmp_path)
    result = deploy_model(cfg, "combined", target="nas5",
                          inference_base=str(tmp_path / "inf"), health_check=False)
    assert result.success, result.message
    assert "nas5" in result.deploy_path


def test_health_check_rejects_incomplete_model(tmp_path):
    from src.deploy import _health_check
    empty = tmp_path / "incomplete"
    empty.mkdir()
    assert _health_check(str(empty)) is False
    assert _health_check(str(tmp_path / "does-not-exist")) is False


def test_get_model_size_sums_files(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 5)
    assert _get_model_size(str(tmp_path)) == 15


# --- multi_deploy fan-out and quorum ---------------------------------------

def _patch_deploy(monkeypatch, ok_nodes: set[str]):
    """Make deploy_model succeed only for ok_nodes; record calls."""
    calls: list[str] = []

    def fake(cfg, label, target="local", inference_base=None, health_check=True,
             rollback=False):
        calls.append(target)
        return _result(target in ok_nodes, node=target)

    monkeypatch.setattr("src.deploy.deploy_model", fake)
    return calls


def test_multi_deploy_no_nodes_fails_closed(monkeypatch):
    from src.deploy import multi_deploy
    results = multi_deploy(make_cfg(), "combined", [])
    assert len(results) == 1
    assert not results[0].success
    assert results[0].message == "no nodes specified"


def test_multi_deploy_expands_all_and_dedupes(monkeypatch, tmp_path):
    from src.deploy import multi_deploy
    os.makedirs(os.path.join(str(tmp_path), "models"), exist_ok=True)
    for node in ("nas5", "laptop"):
        os.makedirs(os.path.join(str(tmp_path), node, "models"), exist_ok=True)
    calls = _patch_deploy(monkeypatch, {"local", "nas5", "laptop"})
    results = multi_deploy(make_cfg(), "combined", ["all", "nas5", "nas5"],
                           inference_base=str(tmp_path))
    assert sorted(r.target for r in results) == ["laptop", "local", "nas5"]
    assert len(calls) == len(set(calls)), "duplicate nodes must be deployed once"


def test_multi_deploy_quorum_counts_partial_success(monkeypatch):
    from src.deploy import multi_deploy
    _patch_deploy(monkeypatch, {"a", "b"})
    results = multi_deploy(make_cfg(), "combined", ["a", "b", "c"], quorum=2)
    assert sum(1 for r in results if r.success) == 2
    assert quorum_met(results, 2) is True


def test_multi_deploy_quorum_unmet_is_reported(monkeypatch, capsys):
    from src.deploy import multi_deploy
    _patch_deploy(monkeypatch, {"a"})
    results = multi_deploy(make_cfg(), "combined", ["a", "b", "c"], quorum=2)
    assert quorum_met(results, 2) is False
    assert "quorum not met" in capsys.readouterr().out


def test_multi_deploy_survives_node_exception(monkeypatch):
    from src.deploy import multi_deploy

    def boom(cfg, label, target="local", inference_base=None, health_check=True,
             rollback=False):
        if target == "bad":
            raise RuntimeError("node unreachable")
        return _result(True, node=target)

    monkeypatch.setattr("src.deploy.deploy_model", boom)
    results = multi_deploy(make_cfg(), "combined", ["good", "bad"])
    by_node = {r.target: r for r in results}
    assert by_node["good"].success is True
    assert by_node["bad"].success is False
    assert "node unreachable" in by_node["bad"].message


def test_multi_deploy_sequential_covers_every_node(monkeypatch):
    from src.deploy import multi_deploy
    calls = _patch_deploy(monkeypatch, {"a", "b"})
    multi_deploy(make_cfg(), "combined", ["a", "b"], parallel=False)
    assert sorted(calls) == ["a", "b"]


def test_discover_nodes_finds_local_and_per_node(monkeypatch, tmp_path):
    from src.deploy import discover_nodes
    os.makedirs(os.path.join(str(tmp_path), "models"), exist_ok=True)
    os.makedirs(os.path.join(str(tmp_path), "nas5", "models"), exist_ok=True)
    os.makedirs(os.path.join(str(tmp_path), "notanode"), exist_ok=True)
    nodes = discover_nodes(make_cfg(), str(tmp_path))
    assert nodes == ["local", "nas5"]

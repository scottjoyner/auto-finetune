"""Temporary-registry custody tests for version and A/B result state.

No actual checkpoints, model loading, network, GPU or production state.
"""
from __future__ import annotations

import json

import pytest

from src import model_registry as registry


class Cfg:
    def __init__(self, root):
        self.root = root
    def path(self, name):
        assert name == "analysis_dir"
        return str(self.root)


def r(tmp_path):
    return registry.ModelRegistry(Cfg(tmp_path))


def add_pair(reg):
    a = reg.register("base", "synthetic-a", "model")
    b = reg.register("test", "synthetic-b", "model")
    return a, b


def test_malformed_existing_registry_ignored_fail_safe(tmp_path):
    path = tmp_path / "model-registry.json"
    path.write_text("{invalid")
    reg = r(tmp_path)
    assert reg.list_versions() == []
    assert reg.active_version() is None


def test_register_and_reload_preserves_candidate(tmp_path):
    reg = r(tmp_path)
    vid = reg.register("candidate", "synthetic", "model", {"quality": .75},
                       notes="synthetic")
    loaded = r(tmp_path).get(vid)
    assert loaded.label == "candidate"
    assert loaded.metrics["quality"] == pytest.approx(.75)
    assert loaded.status == "candidate"
    assert loaded.notes == "synthetic"


def test_missing_id_actions_do_not_mutate_registry(tmp_path):
    reg = r(tmp_path)
    assert reg.get("missing") is None
    assert reg.promote("missing") is False
    assert reg.retire("missing") is False
    assert reg.update_metrics("missing", {"accuracy": 1}) is False
    assert reg.complete_ab_test("missing") is None
    assert reg.get_ab_test("missing") is None
    assert reg.record_ab_metric("missing", "x", "score", 1) is False


def test_list_versions_sorted_and_filtered(tmp_path, monkeypatch):
    clock = iter(range(10, 100))
    monkeypatch.setattr(registry.time, "time", lambda: next(clock))
    reg = r(tmp_path)
    first = reg.register("one", "a", "model")
    second = reg.register("two", "b", "model")
    third = reg.register("three", "c", "model")
    assert [v.version_id for v in reg.list_versions()] == [third, second, first]
    assert len(reg.list_versions(status="candidate")) == 3


def test_update_metrics_retire_and_reload(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    assert reg.update_metrics(a, {"score": .75})
    assert reg.update_metrics(a, {"latency": 1.2})
    assert reg.retire(b)
    reload = r(tmp_path)
    assert reload.get(a).metrics == {"score": .75, "latency": 1.2}
    assert reload.get(b).status == "retired"


def test_registry_summary_tracks_stable_active_and_testing(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    assert reg.promote(a)
    tid = reg.start_ab_test(a, b)
    snapshot = reg.summary()
    assert snapshot["active"] == a
    assert snapshot["total_versions"] == 2
    assert snapshot["testing"] == 1
    assert snapshot["ab_tests_running"] == 1
    assert reg.get(a).status == "active"
    assert reg.get(b).status == "testing"
    assert tid


def test_ab_test_rejects_unknown_participants(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    assert reg.start_ab_test("unknown", b) is None
    assert reg.start_ab_test(a, "unknown") is None
    assert reg.list_ab_tests() == []


def test_ab_metrics_accept_only_running_known_version(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    tid = reg.start_ab_test(a, b)
    assert reg.record_ab_metric(tid, "wrong", "score", .5) is False
    assert reg.record_ab_metric(tid, a, "score", .7) is True
    assert reg.record_ab_metric(tid, a, "score", .8) is True
    assert reg.record_ab_metric(tid, b, "score", .9) is True
    ab = reg.get_ab_test(tid)
    assert ab.metrics_a == {"score": 1.5, "score_count": 2}
    assert ab.metrics_b == {"score": .9, "score_count": 1}


def test_ab_tie_winner_recorded_without_implicit_activation(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    assert reg.promote(a)
    tid = reg.start_ab_test(a, b)
    reg.record_ab_metric(tid, a, "score", .5)
    reg.record_ab_metric(tid, b, "score", .5)
    assert reg.complete_ab_test(tid) == a
    assert reg.complete_ab_test(tid) is None
    assert reg.record_ab_metric(tid, b, "score", .9) is False
    assert reg.active_version().version_id == a
    assert reg.get(b).status == "candidate"


def test_ab_results_reload_without_auto_retirement(tmp_path):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    assert reg.promote(a)
    tid = reg.start_ab_test(a, b)
    assert reg.record_ab_metric(tid, a, "score", .1)
    assert reg.record_ab_metric(tid, b, "score", .9)
    assert reg.complete_ab_test(tid) == b
    new = r(tmp_path)
    assert new.get_ab_test(tid).winner == b
    assert new.get_ab_test(tid).status == "completed"
    assert new.get(a).status == "active"
    assert new.get(b).status == "candidate"
    assert new.summary()["ab_tests_completed"] == 1
    assert new.list_ab_tests(status="completed")[0].test_id == tid


def test_cli_registry_reads_only_temp_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "load", lambda: Cfg(tmp_path))
    assert registry.main(["register", "--label", "test", "--checkpoint",
                          "synthetic-checkpoint"]) == 0
    assert "Registered:" in capsys.readouterr().out
    reg = r(tmp_path)
    vid = reg.list_versions()[0].version_id
    assert registry.main(["list"]) == 0
    assert vid in capsys.readouterr().out
    assert registry.main(["active"]) == 0
    assert "none" in capsys.readouterr().out
    assert registry.main(["summary"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 1


def test_cli_explicit_promote_and_missing_version_returns_status(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "load", lambda: Cfg(tmp_path))
    assert registry.main(["promote", "--version", "missing"]) == 0
    assert "Not found" in capsys.readouterr().out
    reg = r(tmp_path)
    a, _ = add_pair(reg)
    assert registry.main(["promote", "--version", a]) == 0
    assert r(tmp_path).active_version().version_id == a


def test_cli_ab_complete_reports_winner_without_releasing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "load", lambda: Cfg(tmp_path))
    reg = r(tmp_path)
    a, b = add_pair(reg)
    reg.promote(a)
    tid = reg.start_ab_test(a, b)
    assert tid
    assert registry.main(["ab-metric", "--test", tid, "--version", a,
                           "--metric", "score", "--value", "0.7"]) == 0
    assert registry.main(["ab-metric", "--test", tid, "--version", b,
                           "--metric", "score", "--value", "0.9"]) == 0
    assert registry.main(["ab-complete", "--test", tid]) == 0
    assert "Winner:" in capsys.readouterr().out
    assert r(tmp_path).active_version().version_id == a
    assert r(tmp_path).get(b).status == "candidate"


@pytest.mark.parametrize("counts", [(0, 0), (1, 0), (0, 1)])
def test_ab_completion_no_missing_arm_implies_winner(tmp_path, counts):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    reg.promote(a)
    tid = reg.start_ab_test(a, b)
    ab = reg.get_ab_test(tid)
    ab.metrics_a = {"score_count": counts[0], "score": counts[0] * .8}
    ab.metrics_b = {"score_count": counts[1], "score": counts[1] * .9}
    assert reg.complete_ab_test(tid) is None
    assert ab.status == "running"
    assert reg.active_version().version_id == a


def test_ab_cannot_compare_identical_version_to_itself(tmp_path):
    reg = r(tmp_path)
    a, _ = add_pair(reg)
    assert reg.start_ab_test(a, a) is None
    assert reg.list_ab_tests() == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5])
def test_registry_ab_rejects_invalid_metric(tmp_path, bad):
    reg = r(tmp_path)
    a, b = add_pair(reg)
    reg.promote(a)
    tid = reg.start_ab_test(a, b)
    ab = reg.get_ab_test(tid)
    ab.metrics_a = {"score_count": 1, "score": .5}
    ab.metrics_b = {"score_count": 1, "score": bad}
    assert reg.complete_ab_test(tid) is None
    assert reg.active_version().version_id == a
    assert ab.status == "running"

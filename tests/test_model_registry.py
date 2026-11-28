import json

from conftest import make_cfg

from src.model_registry import ABTest, ModelRegistry, ModelVersion, main


def _registry(tmp_path):
    cfg = make_cfg(paths={"analysis_dir": str(tmp_path / "analysis")})
    return ModelRegistry(cfg), cfg


def test_register_get_and_summary(tmp_path):
    reg, _ = _registry(tmp_path)
    vid = reg.register("v1", "/ckpt/v1", "base-model",
                       metrics={"score": 0.5}, notes="first")
    mv = reg.get(vid)
    assert isinstance(mv, ModelVersion)
    assert mv.version_id == vid
    assert mv.label == "v1"
    assert mv.checkpoint_path == "/ckpt/v1"
    assert mv.base_model == "base-model"
    assert mv.metrics == {"score": 0.5}
    assert mv.status == "candidate"
    assert mv.notes == "first"
    assert mv.parent_version is None
    assert reg.get("missing") is None
    assert [v.version_id for v in reg.list_versions()] == [vid]
    summary = reg.summary()
    assert summary["total_versions"] == 1
    assert summary["candidates"] == 1
    assert summary["active"] is None
    assert summary["testing"] == 0
    assert summary["retired"] == 0
    assert summary["ab_tests_running"] == 0
    assert summary["ab_tests_completed"] == 0


def test_registry_persists_and_ignores_corrupt_file(tmp_path):
    reg, cfg = _registry(tmp_path)
    vid = reg.register("persist", "/ckpt/p", "base")
    assert reg.update_metrics(vid, {"score": 0.75}) is True
    registry_path = tmp_path / "analysis" / "model-registry.json"
    assert registry_path.exists()
    saved = json.loads(registry_path.read_text())
    assert [v["version_id"] for v in saved["versions"]] == [vid]
    reopened = ModelRegistry(cfg)
    assert reopened.get(vid).metrics == {"score": 0.75}
    registry_path.write_text("{not json")
    assert ModelRegistry(cfg).list_versions() == []


def test_loads_existing_versions_and_ab_tests(tmp_path):
    cfg = make_cfg(paths={"analysis_dir": str(tmp_path / "analysis")})
    registry_path = tmp_path / "analysis" / "model-registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({
        "versions": [{"version_id": "v1", "label": "v1", "checkpoint_path": "/v1",
                      "base_model": "base", "created_at": 1.0}],
        "ab_tests": [{"test_id": "ab1", "version_a": "v1", "version_b": "v2",
                      "start_time": 1.0}],
        "updated_at": 1.0,
    }))
    reg = ModelRegistry(cfg)
    assert reg.get("v1").version_id == "v1"
    assert reg.get_ab_test("ab1").version_a == "v1"


def test_promote_retire_and_active(tmp_path):
    reg, _ = _registry(tmp_path)
    a = reg.register("a", "/a", "base")
    b = reg.register("b", "/b", "base")
    assert reg.promote("missing") is False
    assert reg.promote(a) is True
    assert reg.active_version().version_id == a
    assert reg.promote(b) is True
    assert reg.get(a).status == "retired"
    assert reg.get(b).status == "active"
    assert [v.version_id for v in reg.list_versions(status="active")] == [b]
    assert [v.version_id for v in reg.list_versions(status="retired")] == [a]
    assert reg.retire("missing") is False
    assert reg.retire(b) is True
    assert reg.active_version() is None


def test_update_metrics_merges_and_rejects_unknown(tmp_path):
    reg, _ = _registry(tmp_path)
    vid = reg.register("metrics", "/m", "base")
    assert reg.update_metrics("missing", {"score": 1.0}) is False
    assert reg.update_metrics(vid, {"score": 0.2}) is True
    assert reg.update_metrics(vid, {"loss": 0.1}) is True
    assert reg.get(vid).metrics == {"score": 0.2, "loss": 0.1}


def test_ab_test_records_metrics_and_promotes_winner(tmp_path):
    reg, _ = _registry(tmp_path)
    a = reg.register("alpha", "/a", "base")
    b = reg.register("beta", "/b", "base")
    assert reg.start_ab_test(a, "missing") is None
    tid = reg.start_ab_test(a, b)
    assert tid is not None
    assert reg.get(a).status == "testing"
    assert reg.get(b).status == "testing"
    assert reg.get(a).ab_test_group == tid
    assert reg.get(b).ab_test_group == tid
    assert reg.list_ab_tests()[0].test_id == tid
    assert [t.test_id for t in reg.list_ab_tests(status="running")] == [tid]
    assert reg.record_ab_metric("missing", a, "score", 1.0) is False
    assert reg.record_ab_metric(tid, "missing", "score", 1.0) is False
    assert reg.record_ab_metric(tid, a, "score", 0.4) is True
    assert reg.record_ab_metric(tid, a, "score", 0.2) is True
    assert reg.record_ab_metric(tid, b, "score", 0.9) is True
    ab = reg.get_ab_test(tid)
    assert isinstance(ab, ABTest)
    assert ab.metrics_a["score_count"] == 2
    assert round(ab.metrics_a["score"], 1) == 0.6
    assert ab.metrics_b == {"score": 0.9, "score_count": 1}
    winner = reg.complete_ab_test(tid, metric="score")
    assert winner == b
    assert reg.get_ab_test(tid).status == "completed"
    assert reg.get_ab_test(tid).winner == b
    assert reg.active_version().version_id == b
    assert reg.get(a).status == "retired"
    assert reg.complete_ab_test(tid) is None
    assert [t.test_id for t in reg.list_ab_tests(status="completed")] == [tid]
    summary = reg.summary()
    assert summary["ab_tests_running"] == 0
    assert summary["ab_tests_completed"] == 1


def test_ab_test_tie_prefers_a_and_missing_completion(tmp_path):
    reg, _ = _registry(tmp_path)
    a = reg.register("a", "/a", "base")
    b = reg.register("b", "/b", "base")
    tid = reg.start_ab_test(a, b)
    assert reg.complete_ab_test("missing") is None
    assert reg.complete_ab_test(tid) == a
    assert reg.get(a).status == "active"
    assert reg.get(b).status == "retired"


def test_cli_dispatch_uses_local_registry(tmp_path, monkeypatch, capsys):
    cfg = make_cfg(paths={"analysis_dir": str(tmp_path / "analysis")})
    monkeypatch.setattr("src.model_registry.load", lambda: cfg)

    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out
    assert main(["active"]) == 0
    assert "Active: none" in capsys.readouterr().out
    assert main(["list"]) == 0
    capsys.readouterr()

    assert main(["register", "--label", "cli-a", "--checkpoint", "/a",
                 "--metrics", '{"score": 0.5}']) == 0
    assert "[registry] Registered:" in capsys.readouterr().out
    assert main(["register", "--label", "cli-b", "--checkpoint", "/b"]) == 0
    capsys.readouterr()
    reg = ModelRegistry(cfg)
    versions = [v.version_id for v in reg.list_versions()]
    a, b = versions[0], versions[1]

    assert main(["list"]) == 0
    assert a in capsys.readouterr().out
    assert main(["promote", "--version", a]) == 0
    assert "[registry] Promoted:" in capsys.readouterr().out
    assert main(["promote", "--version", "missing"]) == 0
    assert "Not found: missing" in capsys.readouterr().out
    assert main(["active"]) == 0
    assert a in capsys.readouterr().out
    assert main(["summary"]) == 0
    assert '"total_versions": 2' in capsys.readouterr().out

    assert main(["ab-test", "--a", a, "--b", b]) == 0
    started = capsys.readouterr().out
    assert "A/B test started:" in started
    tid = started.split("A/B test started: ")[1].strip()
    assert main(["ab-test", "--a", a, "--b", "missing"]) == 0
    assert "Failed to start A/B test" in capsys.readouterr().out
    assert main(["ab-metric", "--test", tid, "--version", b,
                 "--metric", "score", "--value", "0.9"]) == 0
    assert "Metric recorded" in capsys.readouterr().out
    assert main(["ab-metric", "--test", "missing", "--version", a,
                 "--metric", "score", "--value", "0.1"]) == 0
    assert "Failed to record metric" in capsys.readouterr().out
    assert main(["ab-complete", "--test", tid, "--metric", "score"]) == 0
    assert "Winner:" in capsys.readouterr().out
    assert main(["ab-complete", "--test", "missing"]) == 0
    assert "Failed to complete test" in capsys.readouterr().out

"""Disposable registry persistence and A/B state tests.

All model checkpoint paths are inert strings: no model is downloaded,
loaded, trained, activated in a real service, or made available to users.
"""
import json

from src import model_registry as registry
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path)}})


def new(tmp_path):
    return registry.ModelRegistry(cfg(tmp_path))


def test_registration_persistence_and_active_transition(tmp_path,monkeypatch):
    tick=iter(range(100,220))
    monkeypatch.setattr(registry.time,"time",lambda:float(next(tick)))
    r=new(tmp_path)
    v1=r.register("synthetic-a","/nonexistent/a","synthetic-base",{"score":.4},
                  notes="research only")
    v2=r.register("synthetic-b","/nonexistent/b","synthetic-base",{"score":.8},
                  parent=v1)
    assert r.summary()["total_versions"]==2
    assert r.active_version() is None
    assert r.get(v2).parent_version==v1
    assert r.get(v1).metrics=={"score":.4}
    assert r.promote("not-registered") is False
    assert r.promote(v1) is True
    assert r.promote(v2) is True
    assert r.get(v1).status=="retired"
    assert r.active_version().version_id==v2
    reloaded=new(tmp_path)
    assert reloaded.active_version().version_id==v2
    assert reloaded.update_metrics(v2,{"heldout":.72}) is True
    assert reloaded.update_metrics("absent",{"score":1}) is False
    assert new(tmp_path).get(v2).metrics["heldout"]==.72
    assert reloaded.retire("absent") is False
    assert reloaded.retire(v2) is True
    assert new(tmp_path).active_version() is None
    assert len(new(tmp_path).list_versions(status="retired"))==2


def test_robust_empty_corrupt_registry_is_local_only(tmp_path):
    path=tmp_path/"model-registry.json"
    path.write_text("not valid json")
    r=new(tmp_path)
    assert r.summary()["total_versions"]==0
    assert r.list_versions()==[]
    assert r.list_ab_tests()==[]


def test_ab_metrics_aggregation_completion_and_replay(tmp_path,monkeypatch):
    tick=iter(range(100,300))
    monkeypatch.setattr(registry.time,"time",lambda:float(next(tick)))
    r=new(tmp_path)
    a=r.register("synthetic-a","/nowhere/a","toy")
    b=r.register("synthetic-b","/nowhere/b","toy")
    assert r.start_ab_test(a,"unknown") is None
    tid=r.start_ab_test(a,b)
    assert tid
    assert r.summary()["ab_tests_running"]==1
    assert r.get(a).status==r.get(b).status=="testing"
    assert r.record_ab_metric("bad-test",a,"score",1)==False
    assert r.record_ab_metric(tid,"bad-version","score",1)==False
    for n in (0.3,0.5):
        assert r.record_ab_metric(tid,a,"score",n)
    for n in (0.7,0.9):
        assert r.record_ab_metric(tid,b,"score",n)
    assert r.get_ab_test(tid).metrics_b["score_count"]==2
    assert len(r.list_ab_tests("running"))==1
    assert r.complete_ab_test(tid,"score")==b
    assert r.active_version().version_id==b
    assert r.get(a).status=="retired"
    assert r.complete_ab_test(tid) is None
    assert r.record_ab_metric(tid,b,"score",1)==False
    persisted=new(tmp_path)
    assert persisted.get_ab_test(tid).status=="completed"
    assert persisted.summary()["ab_tests_completed"]==1
    assert len(persisted.list_ab_tests(status="completed"))==1
    assert persisted.get_ab_test("missing") is None


def test_cli_summary_active_and_list_use_only_temp_registry(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(registry,"load",lambda:cfg(tmp_path))
    r=new(tmp_path)
    monkeypatch.setattr(registry.time,"time",lambda:100.0)
    rid=r.register("test-label","/fake/adapter","toy")
    assert registry.main(["list"])==0
    assert rid in capsys.readouterr().out
    assert registry.main(["active"])==0
    assert "none" in capsys.readouterr().out
    assert registry.main(["summary"])==0
    assert json.loads(capsys.readouterr().out)["total_versions"]==1
    assert registry.main(["promote","--version","absent"])==0
    assert "Not found" in capsys.readouterr().out
    assert registry.main([])==0
    assert "usage:" in capsys.readouterr().out.lower()


def test_registry_persistence_does_not_open_checkpoint_paths(tmp_path,monkeypatch):
    # Paths are untrusted metadata and MUST NOT be inspected or deserialized
    # by this storage-only test.
    path="/not/a/real/model-with-special-characters"
    r=new(tmp_path)
    id=r.register("metadata-only",path,"toy")
    assert new(tmp_path).get(id).checkpoint_path==path

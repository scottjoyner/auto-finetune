"""Synthetic auto-promotion state-machine tests, no production registries/models."""
import json
import time

from src import auto_promote as promotion
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path)}})


def promoter(tmp_path,**opts):
    r=promotion.AutoPromoter(cfg(tmp_path),
        promotion.PromotionConfig(min_feedback_samples=opts.get("min_feedback_samples",2),
            ab_test_duration_hours=opts.get("ab_test_duration_hours",24.0),
            min_score_threshold=opts.get("min_score_threshold",.6),
            auto_rollback=opts.get("auto_rollback",True)))
    r.alerts.fire=lambda *a,**k:"offline synthetic alert"
    return r


def test_no_ab_test_is_noop(tmp_path):
    p=promoter(tmp_path)
    assert p.check_promotion()=={"action":"none","details":"No active A/B test"}
    assert p.status()["registry_summary"]["total_versions"]==0


def test_registration_with_existing_synthetic_active_creates_ab_test(tmp_path):
    p=promoter(tmp_path)
    old=p.registry.register("toy-old","/invalid/old","toy")
    p.registry.promote(old)
    new=p.register_and_test("toy-new","/invalid/new","toy",{"score":.8})
    tid=p.state["active_ab_test"]
    assert tid
    ab=p.registry.get_ab_test(tid)
    assert {ab.version_a,ab.version_b}=={old,new}
    assert ab.status=="running"
    assert p.registry.get(new).status=="testing"
    assert p.registry.get(old).status=="testing"
    assert promotion.AutoPromoter(cfg(tmp_path)).state["active_ab_test"]==tid


def test_stale_ab_pointer_fails_to_promote(tmp_path):
    p=promoter(tmp_path)
    p.state["active_ab_test"]="no-such-test"
    p._save_state()
    result=p.check_promotion()
    assert result["action"]=="none"
    assert result["details"]=="A/B test not running"
    assert p.state["active_ab_test"] is None
    assert p.registry.active_version() is None


def test_feedback_threshold_and_duration_both_block_early_promotion(tmp_path):
    p=promoter(tmp_path)
    a=p.registry.register("toy-a","/fake/a","toy")
    p.registry.promote(a)
    b=p.register_and_test("toy-b","/fake/b","toy")
    tid=p.state["active_ab_test"]
    assert "Insufficient feedback" in p.check_promotion()["details"]
    p.record_inference(a,"test-session-a","input","output",.7)
    p.record_inference(b,"test-session-b","input","output",.8)
    p.record_inference(b,"test-session-c","input","output",.8)
    assert "Test still running" in p.check_promotion()["details"]
    assert p.registry.get_ab_test(tid).status=="running"
    assert p.registry.get_ab_test(tid).metrics_b["score_count"]==2


def test_winner_is_recorded_after_sufficient_synthetic_evidence(tmp_path):
    p=promoter(tmp_path)
    a=p.registry.register("toy-a","/fake/a","toy")
    p.registry.promote(a)
    b=p.register_and_test("toy-b","/fake/b","toy")
    for name,score in [(a,.5),(a,.5),(b,.9),(b,.9)]:
        p.record_inference(name,"session","prompt","synthetic response",score)
    result=p.check_promotion()
    assert result["action"]=="promoted"
    assert p.registry.active_version().version_id==b
    assert p.state["active_ab_test"] is None
    assert p.state["last_promotion"]>0
    assert promotion.AutoPromoter(cfg(tmp_path)).registry.active_version().version_id==b


def test_below_threshold_winner_rolls_back_in_synthetic_registry(tmp_path):
    p=promoter(tmp_path,min_score_threshold=.9)
    a=p.registry.register("toy-a","/fake/a","toy")
    p.registry.promote(a)
    b=p.register_and_test("toy-b","/fake/b","toy")
    for name,score in [(a,.3),(a,.3),(b,.5),(b,.5)]:
        p.record_inference(name,"session","prompt","synthetic response",score)
    result=p.check_promotion()
    assert result["action"]=="rollback"
    assert p.registry.active_version().version_id==a
    assert p.state["last_rollback"]>0


def test_auto_promotion_state_corruption_is_handled_in_disposable_dir(tmp_path):
    path=tmp_path/"auto-promote-state.json"
    path.write_text("{truncated")
    p=promoter(tmp_path)
    assert p.state=={"active_ab_test":None,"last_promotion":0,"last_rollback":0}


def test_missing_live_model_is_not_exercised_here(tmp_path):
    # A registered path is only metadata; nothing tries to open a checkpoint.
    p=promoter(tmp_path)
    assert p.registry.active_version() is None
    assert not (tmp_path/"model.safetensors").exists()


def test_status_and_check_cli_do_not_contact_external_models(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(promotion,"load",lambda:cfg(tmp_path))
    assert promotion.main(["--check"])==0
    assert "No active A/B test" in capsys.readouterr().out
    assert promotion.main(["--status"])==0
    output=json.loads(capsys.readouterr().out)
    assert output["registry_summary"]["total_versions"]==0
    assert promotion.main(["--register"])==1
    assert "required" in capsys.readouterr().out
    assert promotion.main([])==0
    assert "usage:" in capsys.readouterr().out.lower()


def test_no_active_metric_can_mutate_closed_ab_test(tmp_path):
    p=promoter(tmp_path)
    p.record_inference("synthetic-version","local-test","input","output",score=.2)
    assert p.registry.summary()["total_versions"]==0
    assert p.state["active_ab_test"] is None
    assert p.feedback.summary()["total_entries"]==1
    assert p.check_promotion()["action"]=="none"

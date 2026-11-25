"""Cost accounting on disposable local data; no GPU or cloud billing."""
import json
from pathlib import Path
import pytest
from src import cost
from src.config import Config


def config(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path/"analysis")},
                       "train":{"output_dir":str(tmp_path/"outputs")}})


def test_cost_record_summary_and_reload(tmp_path,monkeypatch):
    tracker=cost.CostTracker(str(tmp_path/"costs"))
    monkeypatch.setattr(cost.time,"time",lambda:100.0)
    tracker.record(label="toy-a",training_hours=1.5,eval_hours=.5,cost_usd=0.0,notes="offline")
    monkeypatch.setattr(cost.time,"time",lambda:200.0)
    tracker.record(label="toy-b",training_hours=4,eval_hours=1,cost_usd=1.25)
    reloaded=cost.CostTracker(str(tmp_path/"costs"))
    assert reloaded.get_summary()["count"]==2
    assert reloaded.get_summary()["total_hours"]==7.0
    assert reloaded.get_summary()["total_cost_usd"]==1.25
    assert reloaded.get_summary("toy-a")["count"]==1
    assert reloaded.get_summary("missing")=={"count":0}
    assert reloaded.get_by_label()["toy-b"]["runs"]==1
    assert reloaded.get_by_label()["toy-a"]["total_hours"]==2.0


def test_newest_cost_history_is_selected_before_limiting(tmp_path,monkeypatch):
    tracker=cost.CostTracker(str(tmp_path/"costs"))
    for i in (100,200,300):
        monkeypatch.setattr(cost.time,"time",lambda i=i:float(i))
        tracker.record(label="toy",training_hours=1.0)
    history=tracker.get_history(limit=2)
    assert [h.timestamp for h in history]==[300.0,200.0]
    assert len(tracker.get_history(label="missing"))==0


def test_disk_usage_only_counts_local_test_files(tmp_path):
    (tmp_path/"checkpoints").mkdir()
    (tmp_path/"checkpoints"/"tiny.bin").write_bytes(b"x"*1024)
    assert cost._get_disk_usage(str(tmp_path/"checkpoints"))==pytest.approx(1/1024)


def test_cli_records_and_lists_only_local_costs(tmp_path,capsys):
    cfg=config(tmp_path)
    assert cost.main(cfg,["cli","cost-record"])==2
    assert cost.main(cfg,["cli","cost-record","--label=toy","--hours=-3"])==2
    assert cost.main(cfg,["cli","cost-record","--label=toy",
                          "--hours=1.25","--eval-hours=.5","--gpu=synthetic"])==0
    assert "recorded" in capsys.readouterr().out
    assert cost.main(cfg,["cli","cost-summary"])==0
    assert "1 runs" in capsys.readouterr().out
    assert cost.main(cfg,["cli","cost-history","--limit=3"])==0
    assert "toy" in capsys.readouterr().out
    assert cost.main(cfg,["cli","cost-history","--limit=0"])==2
    assert cost.main(cfg,["cli","unknown"])==0
    assert "Commands:" in capsys.readouterr().out


def test_cli_checkpoint_size_only_looks_in_disposable_output(tmp_path,capsys):
    cfg=config(tmp_path)
    path=tmp_path/"outputs"/"toolcall-v5-3b-toy"
    path.mkdir(parents=True)
    (path/"fake.bin").write_bytes(b"x"*(1024*1024))
    assert cost.main(cfg,["cli","cost-record","--label=toy","--hours=2"])==0
    assert "checkpoints: 1MB" in capsys.readouterr().out

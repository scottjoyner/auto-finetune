"""Local feedback repair and reliability contracts: no external training calls."""
import json
from pathlib import Path

from src import feedback_loop as feedback
from src.config import Config


def cfg(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path)}})


def test_low_quality_feedback_is_idempotent_and_returns_session_ids(tmp_path):
    collector=feedback.FeedbackCollector(cfg(tmp_path))
    first=collector.record("sess-A","v1","prompt","Error",score=.1,
                           feedback_type="error",expected_text="correct")
    collector.record("sess-A","v1","prompt","Error",score=.2)
    collector.record("sess-B","v1","prompt","valid response",score=.9)
    report=collector.analyze()
    assert report.total_entries==3
    assert report.low_quality_count==2
    assert report.repair_pairs==1
    assert "sess-A" in report.flagged_sessions
    assert "sess-B" not in report.flagged_sessions
    assert collector.get_flagged_sessions()==["sess-A"]
    before=[list(e.issues) for e in collector.entries]
    collector.analyze()
    assert [e.issues for e in collector.entries]==before
    assert collector.summary()["total_entries"]==3
    assert len((tmp_path/"feedback.jsonl").read_text().splitlines())==3
    assert collector.generate_repair_pairs(str(tmp_path/"repairs.jsonl"))==1
    repair=json.loads((tmp_path/"repairs.jsonl").read_text().strip())
    assert repair["input"]=="prompt"
    assert repair["good_output"]=="correct"
    assert first.startswith("fb-")


def test_feedback_reloads_and_truncates_untrusted_inputs(tmp_path):
    c=feedback.FeedbackCollector(cfg(tmp_path))
    c.record("sess","v2","a"*2200,"b"*2200,score=.8,expected_text="c"*2200)
    loaded=feedback.FeedbackCollector(cfg(tmp_path))
    assert len(loaded.entries)==1
    item=loaded.entries[0]
    assert len(item.input_text)==len(item.output_text)==len(item.expected_text)==2000
    assert loaded.analyze().avg_score==.8


def test_detect_repetition_and_error_without_issue_duplication(tmp_path):
    c=feedback.FeedbackCollector(cfg(tmp_path))
    c.record("sess","v2","prompt","exception "*15,score=.0,issues=["low_score"])
    e=c.entries[0]
    assert set(c._detect_issues(e))=={"low_score","error_in_output","repetitive"}
    c.analyze()
    c.analyze()
    assert e.issues.count("low_score")==1
    assert e.issues.count("error_in_output")==1


def test_empty_feedback_and_no_repairs(tmp_path):
    c=feedback.FeedbackCollector(cfg(tmp_path))
    assert c.summary()["total_entries"]==0
    assert c.get_flagged_sessions()==[]
    assert c.generate_repair_pairs(str(tmp_path/"nothing.jsonl"))==0
    assert (tmp_path/"nothing.jsonl").read_text()==""


def test_feedback_cli_analyze_repair_and_flagged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(feedback,"load",lambda:cfg(tmp_path))
    c=feedback.FeedbackCollector(cfg(tmp_path))
    c.record("local-session","v1","prompt","failed",score=.2,expected_text="working")
    out=tmp_path/"repair.jsonl"
    assert feedback.main(["--analyze","--flagged","--repair-pairs",str(out)])==0
    printed=capsys.readouterr().out
    assert "local-session" in printed
    assert "Generated 1 repair pairs" in printed
    assert feedback.main([])==0
    assert "usage:" in capsys.readouterr().out.lower()

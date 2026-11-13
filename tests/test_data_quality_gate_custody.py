"""Real scoring and disposable JSONL quality-gate acceptance evidence.

All input/output paths belong to pytest tmp_path. No fleet data, model,
credentials, inference endpoints, or production sessions are accessed.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from src import data_quality as dq


def good(sid="session-good", source="opencode", content="hello"):
    return {
        "session_id": sid,
        "source": source,
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": content}]},
        ],
    }


def formatted(sid="formatted"):
    return {
        "session_id": sid,
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return path


def test_missing_messages_is_rejected_with_missing_field_issue():
    result = dq.score_session({"session_id": "empty"})
    assert result.passed is False
    assert "no_messages" in result.issues
    assert any("missing_fields:messages" in x for x in result.issues)
    assert result.completeness == 0


@pytest.mark.parametrize("messages", [
    [{"role": "user"}, {"role": "assistant"}],
    [{"role": "user", "parts": [{"text": "hi"}]}, {"parts": [{"text": "bye"}]}],
    ["synthetic", {"role": "assistant", "parts": []}],
])
def test_malformed_message_parts_rejected_or_flagged(messages):
    row = {"session_id": "bad", "source": "opencode", "messages": messages}
    result = dq.score_session(row)
    assert "malformed_message" in result.issues
    assert result.validity < 1


@pytest.mark.parametrize("source", ["unknown", "", None])
def test_unsupported_sources_are_not_silently_valid(source):
    row = good(source=source)
    result = dq.score_session(row)
    assert any("unknown_source" in issue for issue in result.issues)
    assert result.consistency < 1


@pytest.mark.parametrize(("days", "expected"), [
    (1, 1.0), (110, .7), (500, .3),
])
def test_old_training_examples_have_lower_freshness(monkeypatch, days, expected):
    current = 1_900_000_000
    monkeypatch.setattr(dq.time, "time", lambda: current)
    row = good()
    row["time_created"] = current - days * 86400
    assert dq.score_session(row).freshness == expected


def test_raw_validity_including_tool_part_is_checked():
    row = good()
    row["messages"][1]["parts"] = [{"type": "tool", "name": "synthetic"}]
    assert dq.score_session(row).validity == 1


def test_formatted_sessions_are_accepted_without_source():
    score = dq.score_session(formatted())
    assert score.completeness == 1
    assert score.passed


def test_batch_empty_returns_zero_report():
    report = dq.assess_batch([])
    assert report.total == 0
    assert report.passed == 0
    assert report.failed == 0
    assert report.avg_score == 0


def test_batch_duplicate_and_failed_sessions_are_counted():
    rows = [good("one"), good("two"), {"messages": [], "session_id": "invalid"}]
    summary = dq.assess_batch(rows)
    assert summary.total == 3
    assert summary.passed + summary.failed == 3
    assert summary.failed >= 1
    assert summary.issues_summary["duplicate"] >= 1
    assert summary.issues_summary["no_messages"] == 1


def test_batch_invalid_duplicated_message_payload_rejected():
    rows = [good("one"), good("two")]
    report = dq.assess_batch(rows)
    assert report.issues_summary.get("duplicate") == 1


def test_quality_filter_respects_threshold_and_rejects_low_quality():
    rows = [good(), {"session_id": "bad", "messages": []}]
    assert dq.filter_quality(rows, min_score=.6) == [rows[0]]
    assert dq.filter_quality(rows, min_score=1.0) == []


def test_quality_gate_writes_only_passing_synthetic_rows(tmp_path):
    rows = [good(), {"session_id": "bad", "messages": []}]
    inp = write_rows(tmp_path / "source.jsonl", rows)
    output = tmp_path / "out" / "filtered.jsonl"
    result = dq.quality_gate(None, str(inp), str(output))
    assert result["total"] == 2
    assert result["passed"] + result["failed"] == 2
    assert result["output"] == str(output)
    saved = [json.loads(x) for x in output.read_text().splitlines()]
    assert saved == [rows[0]]


def test_quality_gate_rejects_malformed_line_before_any_output(tmp_path):
    inp = tmp_path / "with-malformed-line.jsonl"
    inp.write_text(json.dumps(good()) + "\nnot json\n")
    out = tmp_path / "filtered.jsonl"
    with pytest.raises(ValueError, match="quality_input_malformed"):
        dq.quality_gate(None, str(inp), str(out))
    assert not out.exists()


def test_quality_gate_all_poor_records_are_hold_without_output(tmp_path):
    inp = write_rows(tmp_path / "all-poor.jsonl",
                     [{"session_id": "bad", "messages": []}])
    out = tmp_path / "filtered.jsonl"
    with pytest.raises(ValueError, match="quality_no_passing_records"):
        dq.quality_gate(None, str(inp), str(out))
    assert not out.exists()


def test_quality_cli_report_and_isolated_output(tmp_path, capsys):
    source = write_rows(tmp_path / "records.jsonl", [good(), formatted()])
    target = tmp_path / "filtered" / "out.jsonl"
    assert dq.main(["--input", str(source), "--output", str(target),
                    "--min-score", "0.6", "--report"]) == 0
    stdout = capsys.readouterr().out
    assert "Total: 2" in stdout
    assert "Wrote 2" in stdout
    assert target.is_file()


def test_quality_cli_without_output_only_reports(tmp_path, capsys):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    assert dq.main(["--input", str(inp)]) == 0
    assert "Total: 1" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["valid.jsonl"]


def test_quality_cli_malformed_line_is_hold(tmp_path, capsys):
    inp = tmp_path / "not-all-valid.jsonl"
    inp.write_text("not-json\n" + json.dumps(good()) + "\n")
    assert dq.main(["--input", str(inp)]) == 78
    assert "HOLD" in capsys.readouterr().out


@pytest.mark.parametrize("invalid", [float("nan"), -1, 2, float("inf")])
def test_invalid_quality_threshold_holds_without_output(tmp_path, invalid):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="quality_threshold_invalid"):
        dq.quality_gate(None, str(inp), str(out), min_score=invalid)
    assert not out.exists()
    assert dq.main(["--input", str(inp), "--output", str(out),
                    "--min-score", str(invalid)]) == 78
    assert not out.exists()


def test_quality_gate_cannot_overwrite_preexisting_output(tmp_path):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    out = tmp_path / "existing.jsonl"
    out.write_text("OWNER-DATA")
    with pytest.raises(ValueError, match="quality_output_already_exists"):
        dq.quality_gate(None, str(inp), str(out))
    assert out.read_text() == "OWNER-DATA"


def test_quality_cli_does_not_overwrite_existing_output(tmp_path, capsys):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    out = tmp_path / "existing.jsonl"
    out.write_text("OWNER-DATA")
    assert dq.main(["--input", str(inp), "--output", str(out)]) == 78
    assert "HOLD" in capsys.readouterr().out
    assert out.read_text() == "OWNER-DATA"


def test_quality_gate_empty_input_holds_before_publish(tmp_path):
    inp = tmp_path / "empty.jsonl"
    inp.write_text("\n")
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="quality_input_empty"):
        dq.quality_gate(None, str(inp), str(out))
    assert not out.exists()


def test_quality_publish_refuses_existing_symlink(tmp_path):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    owner = tmp_path / "owner.jsonl"
    owner.write_text("OWNER")
    out = tmp_path / "output.jsonl"
    out.symlink_to(owner)
    with pytest.raises(ValueError, match="quality_output_already_exists"):
        dq.quality_gate(None, str(inp), str(out))
    assert out.is_symlink()
    assert owner.read_text() == "OWNER"


def test_quality_publish_failed_link_cleans_private_stage(tmp_path, monkeypatch):
    inp = write_rows(tmp_path / "valid.jsonl", [good()])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(dq.os, "link", lambda *a, **kw: (_ for _ in ()).throw(OSError("unavailable")))
    with pytest.raises(OSError):
        dq.quality_gate(None, str(inp), str(out))
    assert not out.exists()
    assert not list(tmp_path.glob(".quality-stage-*"))

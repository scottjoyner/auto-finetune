"""Tests for src.extract_opencode and src.extract_hermes."""
from __future__ import annotations

import copy
import json

from conftest import make_cfg

from src.config import _DEFAULTS, Config
from src.extract_hermes import _adapt_record
from src.extract_hermes import main as hermes_main
from src.extract_opencode import _build_part, _safe_json, extract_db, main


def test_safe_json():
    assert _safe_json('{"a":1}') == {"a": 1}
    assert _safe_json("not json") is None
    assert _safe_json("") is None


def test_build_part_text():
    assert _build_part({"type": "text", "text": "hi"}) == {"type": "text", "text": "hi"}


def test_build_part_reasoning():
    assert _build_part({"type": "reasoning", "text": "t"}) == {"type": "reasoning", "text": "t"}


def test_build_part_tool():
    d = {"type": "tool", "tool": "edit", "callID": "c1",
         "state": {"status": "completed", "input": {"a": 1}, "output": "ok"}}
    p = _build_part(d)
    assert p["tool"] == "edit" and p["call_id"] == "c1"
    assert p["input"] == {"a": 1} and p["output"] == "ok"


def test_build_part_patch():
    p = _build_part({"type": "patch", "hash": "h", "files": ["a.py"]})
    assert p["hash"] == "h" and p["files"] == ["a.py"]


def test_build_part_unknown_marker():
    p = _build_part({"type": "step-finish", "reason": "stop", "tokens": {}})
    assert p["type"] == "step-finish"
    assert p["reason"] == "stop"


def test_extract_db_writes_session(make_opencode_db, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg()
    written = extract_db(cfg, make_opencode_db(), str(out))
    assert written == 1
    rec = json.loads((out / "ses_demo.json").read_text())
    assert rec["source"] == "opencode"
    assert rec["title"] == "Demo"
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant"]
    # parts attached to assistant
    asst = [m for m in rec["messages"] if m["role"] == "assistant"][0]
    types = [p["type"] for p in asst["parts"]]
    assert "tool" in types and "text" in types


def test_extract_db_with_broken_tail(make_opencode_db, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    cfg = make_cfg()
    written = extract_db(cfg, make_opencode_db(corrupt_tail=True), str(out))
    assert written == 1


def test_extract_db_excludes_agent(make_opencode_db, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    raw = copy.deepcopy(_DEFAULTS)
    raw["extract"]["exclude_agents"] = ["build"]
    cfg = Config(raw=raw)
    written = extract_db(cfg, make_opencode_db(), str(out))
    assert written == 0


def test_extract_db_min_messages(make_opencode_db, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    raw = copy.deepcopy(_DEFAULTS)
    raw["extract"]["min_messages"] = 99
    cfg = Config(raw=raw)
    written = extract_db(cfg, make_opencode_db(), str(out))
    assert written == 0


def test_main_no_db_path(cfg, tmp_path):
    # db_path empty -> writes nothing, returns 0
    cfg = make_cfg(
        sources={"opencode": {"db_path": ""}},
        paths={
            "raw_dir": str(tmp_path / "raw"),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "dataset_dir": str(tmp_path / "datasets")})
    assert main(cfg) == 0


def test_main_with_db(make_opencode_db, tmp_path):
    cfg = make_cfg(
        sources={"opencode": {"db_path": make_opencode_db()}},
        paths={
            "raw_dir": str(tmp_path / "raw"),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "dataset_dir": str(tmp_path / "datasets")})
    assert main(cfg) == 1


# ── hermes ───────────────────────────────────────────────────────────────────
def test_hermes_adapt_record_ok():
    rec = _adapt_record({"id": "h1", "title": "T", "messages": [{"x": 1}]})
    assert rec["source"] == "hermes"
    assert rec["session_id"] == "h1"
    assert rec["messages"] == [{"x": 1}]


def test_hermes_adapt_record_no_messages():
    assert _adapt_record({"id": "h1"}) is None


def test_hermes_main_disabled(cfg, tmp_path):
    raw = copy.deepcopy(_DEFAULTS)
    raw["sources"]["hermes"]["enabled"] = False
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_path / "raw"),
        "cleaned_dir": str(tmp_path / "cleaned"),
        "dataset_dir": str(tmp_path / "datasets")})
    assert hermes_main(cfg) == 0


def test_hermes_main_reads_json(tmp_path):
    hermes_dir = tmp_path / "hermes"
    hermes_dir.mkdir()
    (hermes_dir / "s1.json").write_text(json.dumps(
        {"id": "h1", "title": "T", "messages": [{"role": "user"}]}))
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = make_cfg(
        sources={"hermes": {"dir": str(hermes_dir), "enabled": True}},
        paths={
            "raw_dir": str(raw_dir),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "dataset_dir": str(tmp_path / "datasets")})
    n = hermes_main(cfg)
    assert n == 1
    assert (raw_dir / "hermes_h1.json").exists()


def test_hermes_main_reads_jsonl(tmp_path):
    hermes_dir = tmp_path / "hermes"
    hermes_dir.mkdir()
    (hermes_dir / "s.jsonl").write_text(json.dumps(
        {"id": "h1", "messages": [{}]}) + "\n" + json.dumps(
        {"id": "h2", "messages": [{}]}) + "\n")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    cfg = make_cfg(
        sources={"hermes": {"dir": str(hermes_dir), "enabled": True}},
        paths={
            "raw_dir": str(raw_dir),
            "cleaned_dir": str(tmp_path / "cleaned"),
            "dataset_dir": str(tmp_path / "datasets")})
    assert hermes_main(cfg) == 2

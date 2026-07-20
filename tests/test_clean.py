"""Tests for src.clean."""
from __future__ import annotations

import copy

from conftest import make_cfg

from src.clean import (
    _conv_hash,
    _redact_obj,
    clean_message,
    clean_session,
    redact,
)
from src.config import _DEFAULTS, Config


def _cfg(**overrides):
    raw = copy.deepcopy(_DEFAULTS)
    return Config(raw=raw)


def test_redact_api_key():
    out = redact("api_key=abcdef1234567890abcdef")
    assert "abcdef1234567890abcdef" not in out
    assert "REDACTED" in out


def test_redact_openai_sk():
    out = redact("token sk-1234567890abcdefghijklmnopqrstuv")
    assert "REDACTED_OPENAI" in out


def test_redact_github_token():
    out = redact("ghp_1234567890abcdefghijklmnopqrstuv")
    assert "REDACTED_GITHUB" in out


def test_redact_private_key():
    out = redact("-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----")
    assert "REDACTED_KEY" in out


def test_redact_non_string_passthrough():
    assert redact(123) == 123
    assert redact(None) is None


def test_redact_obj_nested():
    obj = {"a": "api_key=abcdef1234567890abcdef", "b": ["password=secretvalue123",
           {"c": "sk-1234567890abcdefghijklmnopqrstuv"}]}
    red = _redact_obj(obj)
    assert "REDACTED" in red["a"]
    assert "REDACTED" in red["b"][0]
    assert "REDACTED_OPENAI" in red["b"][1]["c"]


def test_clean_message_drops_reasoning_by_default():
    cfg = make_cfg()
    msg = {"id": "m", "role": "assistant", "parts": [
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "answer"},
    ]}
    out = clean_message(msg, cfg)
    types = [p["type"] for p in out["parts"]]
    assert "reasoning" not in types
    assert "text" in types


def test_clean_message_keeps_reasoning_when_enabled():
    cfg = make_cfg(clean={"keep_reasoning_as_context": True})
    msg = {"id": "m", "role": "assistant", "parts": [
        {"type": "reasoning", "text": "thinking"},
    ]}
    out = clean_message(msg, cfg)
    assert any(p["type"] == "reasoning" for p in out["parts"])


def test_clean_message_drops_empty_turn():
    cfg = make_cfg()
    msg = {"id": "m", "role": "assistant", "parts": [
        {"type": "step-start"},  # no content
    ]}
    assert clean_message(msg, cfg) is None


def test_clean_message_tool_has_content():
    cfg = make_cfg()
    msg = {"id": "m", "role": "assistant", "parts": [
        {"type": "tool", "tool": "bash", "input": "ls", "output": "ok"},
    ]}
    out = clean_message(msg, cfg)
    assert out is not None


def test_clean_message_truncates_long_text():
    cfg = make_cfg(clean={"max_chars_per_message": 10})
    msg = {"id": "m", "role": "user", "parts": [
        {"type": "text", "text": "x" * 100},
    ]}
    out = clean_message(msg, cfg)
    assert len(out["parts"][0]["text"]) <= 10 + len("\n...[truncated]")


def test_clean_message_no_redact_when_disabled():
    cfg = make_cfg(clean={"redact_secrets": False})
    msg = {"id": "m", "role": "user", "parts": [
        {"type": "text", "text": "api_key=abcdef1234567890abcdef"},
    ]}
    out = clean_message(msg, cfg)
    assert "abcdef1234567890abcdef" in out["parts"][0]["text"]


def test_clean_session_drops_short():
    cfg = make_cfg()
    rec = {"session_id": "s", "messages": [{"id": "m", "role": "user",
           "parts": [{"type": "text", "text": "hi"}]}]}
    assert clean_session(rec, cfg) is None


def test_clean_session_keeps_valid():
    cfg = make_cfg()
    rec = {"session_id": "s", "messages": [
        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"id": "m2", "role": "assistant", "parts": [{"type": "text", "text": "yo"}]},
    ]}
    out = clean_session(rec, cfg)
    assert out is not None and len(out["messages"]) == 2


def test_conv_hash_deterministic():
    a = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]}
    b = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]}
    assert _conv_hash(a) == _conv_hash(b)
    c = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "y"}]}]}
    assert _conv_hash(a) != _conv_hash(c)


def test_main_writes_and_dedupes(tmp_root, sample_session):
    raw = tmp_root / "data" / "raw"
    cleaned = tmp_root / "data" / "cleaned"
    # two identical sessions -> one after dedupe
    import json
    (raw / "a.json").write_text(json.dumps(sample_session))
    (raw / "b.json").write_text(json.dumps(sample_session))
    cfg = make_cfg()
    # repoint paths
    cfg = make_cfg(paths={
        "raw_dir": str(raw), "cleaned_dir": str(cleaned), "dataset_dir": str(tmp_root / "data" / "datasets")})
    written = __import__("src.clean", fromlist=["main"]).main(cfg)
    assert written == 1
    assert len(list(cleaned.glob("*.json"))) == 1

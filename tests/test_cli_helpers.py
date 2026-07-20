"""Coverage tests for pure argument-parsing helpers in cli.py."""
from __future__ import annotations

import src.cli as cli


def test_parse_source():
    assert cli._parse_source(["run", "--source=hermes"]) == "hermes"
    assert cli._parse_source(["run"]) is None


def test_parse_label():
    assert cli._parse_label(["x", "--label=ssd"]) == "ssd"
    assert cli._parse_label(["x"]) is None


def test_parse_int_flag():
    assert cli._parse_int_flag(["--max-examples=100"], "--max-examples") == 100
    assert cli._parse_int_flag(["--max-examples=abc"], "--max-examples") is None
    assert cli._parse_int_flag([], "--max-examples") is None


def test_parse_str_flag():
    assert cli._parse_str_flag(["--project=portfolio"], "--project") == "portfolio"
    assert cli._parse_str_flag([], "--project") is None


def test_has_flag():
    assert cli._has_flag(["--push"], "--push") is True
    assert cli._has_flag([], "--push") is False


def test_local_ref_specs(monkeypatch):
    monkeypatch.setenv("LOCAL_REF_MODEL", "/fake/qwen2.5-7b")
    monkeypatch.setattr("src.cli.os.path.exists", lambda *a, **k: False)
    specs = cli._local_ref_specs()
    assert specs
    assert specs[0]["name"] == "qwen2.5-7b"
    assert specs[0]["model_path"] == "/fake/qwen2.5-7b"
    assert specs[0]["runner"] == "local-chat"

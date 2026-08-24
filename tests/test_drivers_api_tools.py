"""Tests for ApiToolsDriver: structured tool_calls -> bench dialect."""
from __future__ import annotations

import json
import urllib.error

import pytest

from src.drivers_api_tools import ApiToolsDriver


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return json.dumps(self._payload).encode()


def _driver(monkeypatch, message):
    d = ApiToolsDriver(base_url="http://127.0.0.1:1/v1", model="m")
    captured = {}
    def fake_urlopen(req, timeout=600):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResponse({"choices": [{"message": message}]})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return d, captured


def test_structured_calls_translate_to_bench_dialect(monkeypatch):
    msg = {"content": "<think>reasoning</think>",
           "tool_calls": [{"type": "function", "id": "1", "function": {
               "name": "write",
               "arguments": json.dumps({"filePath": "a.txt", "content": "hi"})}}]}
    d, cap = _driver(monkeypatch, msg)
    text = d.generate([{"role": "user", "content": "make a.txt"}])
    assert "<think>" not in text
    assert '<tool_call>' in text and '"name": "write"' in text
    # bench's default parser must recover it
    from src.bench import parse_tool_calls
    calls = parse_tool_calls(text)
    assert calls[0]["name"] == "write"
    assert calls[0]["args"] == {"filePath": "a.txt", "content": "hi"}
    assert cap["body"]["tools"][0]["function"]["name"] in ("bash", "write")


def test_no_calls_returns_clean_content(monkeypatch):
    d, _ = _driver(monkeypatch, {"content": "All done.",
                                 "tool_calls": []})
    assert d.generate([{"role": "user", "content": "status"}]) == "All done."


def test_string_arguments_that_fail_parse_map_to_empty(monkeypatch):
    msg = {"content": "", "tool_calls": [{"type": "function", "function": {
        "name": "bash", "arguments": "not-json"}}]}
    d, _ = _driver(monkeypatch, msg)
    text = d.generate([{"role": "user", "content": "run"}])
    # bench loop uses its default parser on our emitted dialect
    from src.bench import parse_tool_calls as canon
    parsed = canon(text)
    assert parsed[0]["name"] == "bash"
    assert isinstance(parsed[0]["args"], dict)


def test_registration_and_base_url_required():
    from src.bench import RUNNERS
    assert "api-tools" in RUNNERS
    with pytest.raises(ValueError):
        RUNNERS["api-tools"]()

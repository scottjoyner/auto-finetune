"""Tests for src.subagent (MCP/ACP-over-stdio protocol adapter).

All tests use injected fakes for make_driver / run_one, so no GPU or model is
loaded. This exercises the JSON-RPC handler: initialize, tools/list, tools/call,
notifications, and a full stdio request/response exchange.
"""
from __future__ import annotations

import io
import json
from unittest.mock import patch

from src import subagent as S
from src.bench import TaskResult


class FakeDriver:
    def __init__(self, *a, **k):
        pass


def _ctx():
    def make_driver(model_path, variant):
        return FakeDriver()

    def run_one(driver, task, model_name, runner_name, sandbox_root=None,
                gen_max_tokens=512):
        return TaskResult(task_id=task.id, kind=task.kind, model=model_name,
                          runner=runner_name, completed=True, turns=2)

    return S.SubagentContext(make_driver=make_driver, run_one=run_one)


def test_initialize():
    resp = S.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                             "params": {}}, _ctx())
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == S.SERVER_NAME
    assert "tools" in resp["result"]["capabilities"]


def test_notifications_initialized_returns_none():
    assert S.handle_request(
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _ctx()) is None


def test_tools_list():
    resp = S.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                            _ctx())
    tools = resp["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "run_task"
    assert "prompt" in tools[0]["inputSchema"]["properties"]


def test_tools_call_success():
    req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
           "params": {"name": "run_task",
                      "arguments": {"prompt": "do x",
                                    "model_path": "/m/model",
                                    "variant": "base"}}}
    resp = S.handle_request(req, _ctx())
    assert resp["id"] == 3
    text = resp["result"]["content"][0]["text"]
    assert "completed=True" in text
    assert "turns=2" in text


def test_tools_call_missing_args_errors():
    req = {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
           "params": {"name": "run_task", "arguments": {"prompt": "x"}}}
    resp = S.handle_request(req, _ctx())
    assert "error" in resp
    assert resp["error"]["code"] == -32602


def test_unknown_method_errors():
    resp = S.handle_request({"jsonrpc": "2.0", "id": 5, "method": "bogus"},
                            _ctx())
    assert resp["error"]["code"] == -32601


def test_stdio_roundtrip():
    lines = [
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
        '{"jsonrpc":"2.0","method":"notifications/initialized"}',
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}',
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":'
        '{"name":"run_task","arguments":{"prompt":"hi","model_path":"/m"}}}',
    ]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
        try:
            S.serve_stdio("/m/model", variant="auto", rocm=False, ctx=_ctx())
        except Exception:
            pass
    out = stdout.getvalue()
    responses = [json.loads(l) for l in out.splitlines() if l.strip()]
    ids = {r.get("id") for r in responses}
    assert 1 in ids and 2 in ids and 3 in ids
    call_resp = next(r for r in responses if r.get("id") == 3)
    assert "completed=True" in call_resp["result"]["content"][0]["text"]

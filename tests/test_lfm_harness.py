"""Tests for src.lfm_harness (MCP stdio adapter) — pure handler, fake driver."""
from __future__ import annotations

import json
import sys

from src import lfm_harness as H
from src.bench import TaskResult


class FakeDriver:
    def __init__(self, **kw):
        self.kw = kw


def _ctx(completed=True, turns=2):
    captured = {}

    def make_driver(**kw):
        captured.update(kw)
        return FakeDriver(**kw)

    def run_one(driver, task, model_name, runner_name, sandbox_root=None,
                gen_max_tokens=512):
        return TaskResult(task_id=task.id, kind=task.kind, model=model_name,
                          runner=runner_name, completed=completed,
                          turns=turns)
    return H.HarnessContext(make_driver=make_driver, run_one=run_one), captured


def test_initialize():
    ctx, _ = _ctx()
    resp = H.handle_request({"jsonrpc": "2.0", "id": 1,
                             "method": "initialize"}, ctx)
    assert resp["result"]["serverInfo"]["name"] == "lfm25-subagent"


def test_tools_list():
    ctx, _ = _ctx()
    resp = H.handle_request({"jsonrpc": "2.0", "id": 2,
                             "method": "tools/list"}, ctx)
    tools = resp["result"]["tools"]
    assert [t["name"] for t in tools] == ["lfm_task"]
    assert "prompt" in tools[0]["inputSchema"]["properties"]


def test_tool_call_passes_driver_kwargs():
    ctx, captured = _ctx()
    resp = H.handle_request({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "lfm_task",
                   "arguments": {"prompt": "do a thing",
                                 "max_turns": 5, "temperature": 0.1}}},
        ctx)
    body = resp["result"]["content"][0]["text"]
    assert "completed=True" in body
    assert captured["temperature"] == 0.1
    task = captured  # driver got base_url default
    assert captured.get("base_url", H.DEFAULT_BASE_URL)


def test_unknown_tool_and_method():
    ctx, _ = _ctx()
    r1 = H.handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                           "params": {"name": "nope", "arguments": {}}}, ctx)
    assert r1["error"]["code"] == -32601
    r2 = H.handle_request({"jsonrpc": "2.0", "id": 5,
                           "method": "bogus/thing"}, ctx)
    assert r2["error"]["code"] == -32601


def test_missing_prompt_is_invalid_params():
    ctx, _ = _ctx()
    r = H.handle_request({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                          "params": {"name": "lfm_task",
                                     "arguments": {}}}, ctx)
    assert r["error"]["code"] == -32602


def test_stdio_exchange_roundtrip():
    ctx, _ = _ctx()
    lines = [json.dumps({"jsonrpc": "2.0", "id": 7,
                         "method": "tools/list"}),
             "not json at all",
             json.dumps({"jsonrpc": "2.0", "method":
                         "notifications/initialized"})]
    import io
    monkey_in = io.StringIO("\n".join(lines) + "\n")
    real_stdin, real_stdout = sys.stdin, sys.stdout
    sys.stdin = monkey_in
    out = io.StringIO()
    sys.stdout = out
    try:
        H.serve_stdio(ctx)
    finally:
        sys.stdin, sys.stdout = real_stdin, real_stdout
    responses = [json.loads(l) for l in out.getvalue().splitlines() if l]
    assert len(responses) == 1
    assert responses[0]["id"] == 7

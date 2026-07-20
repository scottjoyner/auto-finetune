"""Tests for src.drivers_localchat (standard HF chat-model driver)."""
from __future__ import annotations

import json
import torch
from pathlib import Path

from src import drivers_localchat as D
from src.bench import run_task, Task, register_runner, make_driver


class _FakeTok:
    pad_token_id = 0

    def apply_chat_template(self, *a, **k):
        return "PROMPT"

    def __call__(self, *a, **k):
        # zero-length prompt so LocalChatDriver's out[0][ids.shape[1]:] slice
        # keeps the full generated token sequence (mimics real HF behavior)
        return type("T", (), {"input_ids": torch.tensor([[]])})()

    def decode(self, ids, skip_special_tokens=False):
        # ids come back as [1, <char codes...>, 2]; strip the 1 and 2
        return "".join(chr(int(i)) for i in ids[1:-1])


class _FakeModel:
    device = "cpu"

    def generate(self, ids, **k):
        txt = '{"name": "write", "arguments": {"filePath": "g.txt", "content": "hi"}}'
        return torch.tensor([[1] + [ord(c) for c in txt] + [2]])


def _fake_local_chat(model_path, **kw):
    return D.LocalChatDriver(model_path, _model=_FakeModel(), _tok=_FakeTok())


def test_parse_native_both_shapes():
    a = D.parse_native_tool_calls('x {"name": "bash", "arguments": {"command": "ls"}} y')
    assert a == [{"name": "bash", "args": {"command": "ls"}}]
    b = D.parse_native_tool_calls(
        '{"function": {"name": "write", "parameters": {"filePath": "a", "content": "x"}}}')
    assert b[0]["name"] == "write"
    assert b[0]["args"]["filePath"] == "a"


def test_parse_native_nested():
    txt = 'call: {"name": "edit", "arguments": {"filePath": "f", "oldText": "a", "newText": "x"}}'
    calls = D.parse_native_tool_calls(txt)
    assert any(c["name"] == "edit" for c in calls)
    assert calls[0]["args"]["newText"] == "x"


def test_parse_native_skips_non_call_blocks():
    # brace blocks without "name"/"function" are skipped (covers the continue)
    txt = 'noise {"random": 1} and {"name": "bash", "arguments": {}} end'
    calls = D.parse_native_tool_calls(txt)
    # only the block carrying a tool name survives; args:{} parsed as empty dict
    assert calls == [{"name": "bash", "args": {}}]
    # a block with no name/function is dropped
    assert D.parse_native_tool_calls('{"random": 1}') == []


def test_parse_native_skips_unparseable_json():
    # a brace block that is not valid JSON yields no calls (covers the except)
    assert D.parse_native_tool_calls('{not valid json{{') == []
    # a balanced block with valid JSON but no name is also skipped
    assert D.parse_native_tool_calls('{"foo": [1,2,3]}') == []


def test_localchat_generate_wraps_tools(monkeypatch):
    # generate() should pass the tool schemas to apply_chat_template and decode
    # the generated ids. Use injected fake model/tok (no real 7B load).
    seen = {}
    class Tok:
        pad_token_id = 0
        def apply_chat_template(self, messages, tools=None, **k):
            seen["tools"] = tools
            return "PROMPT"
        def __call__(self, *a, **k):
            return type("T", (), {"input_ids": torch.tensor([[]])})()
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in ids[1:-1])
    class Mdl:
        device = "cpu"
        def generate(self, ids, **k):
            t = '{"name": "bash", "arguments": {"command": "ls"}}'
            return torch.tensor([[1] + [ord(c) for c in t] + [2]])
    drv = D.LocalChatDriver("/x", _model=Mdl(), _tok=Tok())
    out = drv.generate([{"role": "user", "content": "hi"}], max_new_tokens=16)
    assert '"name": "bash"' in out
    # the tool schemas were forwarded to the chat template
    assert seen["tools"] and any(t["function"]["name"] == "bash" for t in seen["tools"])


def test_make_local_chat_factory():
    drv = D._make_local_chat(model_path="/x")
    assert isinstance(drv, D.LocalChatDriver)
    assert drv.model_path == "/x"
    assert drv.rocm is False


def test_localchat_run_task_writes_file(tmp_path):
    # register the fake factory so make_driver("local-chat") returns it
    register_runner("local-chat", _fake_local_chat)
    drv = make_driver("local-chat", model_path="/x")
    task = Task.from_dict(json.loads(
        '{"id":"t","prompt":"x",'
        '"checks":[{"kind":"file_exists","path":"g.txt"},'
        '{"kind":"file_contains","path":"g.txt","expect":"hi"}]}'))
    res = run_task(drv, task, "qwen", "local-chat", sandbox_root=tmp_path)
    assert res.success
    assert res.checks_passed == 2
    assert (tmp_path / "g.txt").read_text() == "hi"

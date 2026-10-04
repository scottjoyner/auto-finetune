"""Tests for MiniCPM5 endpoint driver, parser, MCP adapter, and harness."""
import json
import tempfile
from pathlib import Path

import pytest

from src import subagent_minicpm5 as M
from src.bench import ModelDriver, Task, make_driver, run_task


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self):
        return json.dumps(self.payload).encode()


class ScriptedDriver(ModelDriver):
    parse_tool_calls = staticmethod(M.parse_minicpm_tool_calls)
    def __init__(self):
        self.n = 0
    def generate(self, messages, max_new_tokens=512):
        self.n += 1
        return ('<function name="bash"><param name="command">echo hi > out.txt</param>'
                if self.n == 1 else "done")


def test_parse_strict_xml():
    t = ('<function name="bash"><param name="command">ls -la</param>')
    calls = M.parse_minicpm_tool_calls(t)
    assert len(calls) == 1 and calls[0]["name"] == "bash"
    assert calls[0]["args"] == {"command": "ls -la"}

def test_parse_xml_cdata():
    t = '<function name="bash"><param name="command"><![CDATA[echo ok]]></param>'
    calls = M.parse_minicpm_tool_calls(t)
    assert calls[0]["args"] == {"command": "echo ok"}

def test_parse_bash_tag():
    t = '<bash command="pwd">'
    calls = M.parse_minicpm_tool_calls(t)
    assert calls == [{"name": "bash", "args": {"command": "pwd"}}]

def test_parse_bash_line():
    t = "bash: echo hello"
    calls = M.parse_minicpm_tool_calls(t)
    assert calls == [{"name": "bash", "args": {"command": "echo hello"}}]

def test_parse_native_json():
    t = '{"name":"bash","arguments":{"command":"ls"}}'
    calls = M.parse_minicpm_tool_calls(t)
    assert calls == [{"name": "bash", "args": {"command": "ls"}}]

def test_parse_empty():
    assert M.parse_minicpm_tool_calls("just prose") == []

def test_make_minicpm_driver():
    drv = M.make_minicpm_driver()
    assert isinstance(drv, M.MiniCPM5Driver)
    assert drv.base_url == M.DEFAULT_BASE_URL


def test_handle_request_tools_list():
    ctx = M.SubagentContext(root=Path(tempfile.mkdtemp()))
    resp = M.handle_request({"jsonrpc":"2.0","id":1,"method":"tools/list"}, ctx)
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "bash" in names and "run_task" in names

def test_handle_request_bash(tmp_path: Path):
    ctx = M.SubagentContext(root=tmp_path)
    resp = M.handle_request({"jsonrpc":"2.0","id":2,"method":"tools/call",
                            "params":{"name":"bash","arguments":{"command":"echo hi"}}}, ctx)
    assert resp["jsonrpc"] == "2.0"
    assert not resp["result"]["isError"]

def test_handle_request_read_write(tmp_path: Path):
    ctx = M.SubagentContext(root=tmp_path)
    M.handle_request({"jsonrpc":"2.0","id":3,"method":"tools/call",
                     "params":{"name":"write","arguments":{"filePath":"t.txt","content":"hello"}}}, ctx)
    resp = M.handle_request({"jsonrpc":"2.0","id":4,"method":"tools/call",
                             "params":{"name":"read","arguments":{"filePath":"t.txt"}}}, ctx)
    assert "hello" in resp["result"]["content"][0]["text"]

def test_handle_request_run_task(monkeypatch):
    ctx = M.SubagentContext()
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"choices":[{"message":{"content":"done"}}]}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: FakeResp())
    resp = M.handle_request({"jsonrpc":"2.0","id":5,"method":"tools/call",
                             "params":{"name":"run_task","arguments":{"prompt":"x"}}}, ctx)
    assert "completed=True" in resp["result"]["content"][0]["text"]

def test_handle_request_unknown_tool():
    ctx = M.SubagentContext()
    resp = M.handle_request({"jsonrpc":"2.0","id":6,"method":"tools/call",
                             "params":{"name":"bogus","arguments":{}}}, ctx)
    assert resp["error"]["code"] == -32601

def test_run_task_with_fake_driver(tmp_path: Path):
    task = Task(id="t", prompt="p", kind="exec",
                checks=[{"kind":"file_exists","path":"out.txt"}])
    drv = ScriptedDriver()
    res = run_task(drv, task, "fake", "minicpm5", sandbox_root=tmp_path)
    assert res.success
    assert (tmp_path / "out.txt").exists()

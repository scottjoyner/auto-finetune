"""Tests for the 2B bash subagent's output parsing.

subagent_2b_agent.py had no tests at all. Its job is to decide what the model
meant to run and then hand that string to subprocess.run(shell=True), so the
tests concentrate on the parsers: a false positive here is an English sentence
executed as a shell command.

call_2b and the agent loop are exercised with a stub model and a stub
subprocess, so nothing is actually executed.
"""
from __future__ import annotations

import subprocess

import pytest

from src import subagent_2b_agent as A

# --- extract_bash ----------------------------------------------------------

def test_extract_bash_xml_function():
    txt = '<function name="bash"><param name="command">ls -la</param></function>'
    assert A.extract_bash(txt) == ["ls -la"]


def test_extract_bash_xml_with_cdata():
    txt = ('<function name="bash"><param name="command">'
           '<![CDATA[echo hi]]></param></function>')
    assert A.extract_bash(txt) == ["echo hi"]


def test_extract_bash_multiple_calls():
    txt = ('<function name="bash"><param name="command">ls</param></function>\n'
           '<function name="bash"><param name="command">df -h</param></function>')
    assert A.extract_bash(txt) == ["ls", "df -h"]


@pytest.mark.parametrize("txt,expected", [
    ('<bash>command="df -h"</bash>', ["df -h"]),
    ("<bash>command='ps aux'</bash>", ["ps aux"]),
    ("thinking\nbash: ls -la\nresult:", ["ls -la"]),
])
def test_extract_bash_alternate_formats(txt, expected):
    assert A.extract_bash(txt) == expected


def test_extract_bash_ignores_non_bash_function():
    assert A.extract_bash('<function name="python"><param name="x">1</param></function>') == []


def test_extract_bash_empty_input():
    assert A.extract_bash("") == []
    assert A.extract_bash("no commands here at all") == []


def test_extract_bash_strips_whitespace():
    txt = '<function name="bash"><param name="command">\n   ls -la  \n</param></function>'
    assert A.extract_bash(txt) == ["ls -la"]


# --- extract_fallback: prose must not become a command --------------------

@pytest.mark.parametrize("label,text", [
    # Each of these contains a short command name as a substring of an English
    # word: calls/ls, islands/ls, campus/ps, pressure/ss.
    ("calls", "Here is my plan.\n```\nThe agent makes 3 calls per turn\n```"),
    ("islands", "```\nfishing in the islands\n```"),
    ("campus", "```\nsee the campus map\n```"),
    ("pressure", "```\ncheck the pressure reading\n```"),
])
def test_prose_is_not_mistaken_for_a_command(label, text):
    # Regression: `kw in cmd` matched these and the result went to shell=True.
    assert A.extract_fallback(text) is None, f"{label} parsed as a command"


def test_real_command_in_backticks_is_found_despite_prose():
    # The false positive used to return early, hiding the real command.
    text = "Plan:\n```\nThe agent makes 3 calls per turn\n```\nNow run: `ls -la /tmp`"
    assert A.extract_fallback(text) == "ls -la /tmp"


@pytest.mark.parametrize("label,text,expected", [
    ("bash block", "Run:\n```bash\nls -la /tmp\n```", "ls -la /tmp"),
    ("plain block", "Run:\n```\ndf -h\n```", "df -h"),
    ("inline backtick", "Try `df -h` first", "df -h"),
    ("sudo prefix", "```\nsudo rm -f /tmp/x\n```", "sudo rm -f /tmp/x"),
])
def test_real_commands_are_still_found(label, text, expected):
    assert A.extract_fallback(text) == expected


def test_leading_comment_line_does_not_hide_command():
    text = "```\n# setup step\nls -la\n```"
    assert A.extract_fallback(text) == "# setup step\nls -la"


def test_dollar_form_is_found():
    # Known limitation, unchanged here: the $ pattern consumes to end of line,
    # so trailing prose is captured too ("$ pwd to check" -> "pwd to check").
    # That fails harmlessly when run, unlike the prose false positives above,
    # and bounding it properly needs command-boundary logic out of scope here.
    got = A.extract_fallback("Type $ pwd to check")
    assert got.startswith("pwd")


def test_fallback_returns_none_for_plain_prose():
    assert A.extract_fallback("I think we should look at the logs.") is None


# --- extract_function ------------------------------------------------------

def test_extract_function_balanced():
    name, code = A.extract_function("backup() {\n  tar czf x.tgz y\n}\n")
    assert name == "backup"
    assert code == "backup() {\n  tar czf x.tgz y\n}"


def test_extract_function_nested_braces():
    name, code = A.extract_function("f() {\n  if true; then\n    g() {\n      echo x\n    }\n  fi\n}\n")
    assert name == "f"
    assert code.count("{") == code.count("}") == 2
    assert code.startswith("f() {") and code.endswith("}")


def test_extract_function_skips_truncated_candidate():
    # Regression: returned on the first name(){ even with unbalanced braces,
    # so a truncated fragment shadowed a valid function later in the output.
    text = ("Here is a fragment:\noops() {\n  echo broken\n\n"
            "And the real one:\nbackup() {\n  echo hi\n}\n")
    name, code = A.extract_function(text)
    assert name == "backup"
    assert "echo hi" in code


def test_extract_function_none_when_all_truncated():
    name, code = A.extract_function("broken() {\n  echo hi\n")
    assert (name, code) == (None, None)


def test_extract_function_none_when_absent():
    assert A.extract_function("just prose") == (None, None)


# --- run_2b_agent loop (stubbed model, stubbed subprocess) -----------------

class _Proc:
    def __init__(self, out="ok\n"):
        self.stdout = out
        self.stderr = ""
        self.returncode = 0


@pytest.fixture
def stub_model(monkeypatch):
    """Feed a scripted sequence of model replies; record nothing executes."""
    replies = []
    calls = {"n": 0, "executed": []}

    def fake_call_2b(messages, model="minicpm5-2b", max_tokens=512, **kw):
        idx = calls["n"]
        calls["n"] += 1
        content = replies[idx] if idx < len(replies) else replies[-1]
        return {"content": content, "tok_s": 50.0, "elapsed": 1.0, "finish": "stop"}

    def fake_run(cmd, **kw):
        calls["executed"].append(cmd)
        return _Proc()

    monkeypatch.setattr(A, "call_2b", fake_call_2b)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls, replies


def test_agent_executes_extracted_tool_call(stub_model):
    calls, replies = stub_model
    replies[:] = ['<function name="bash"><param name="command">ls -la</param></function>']
    out = A.run_2b_agent("list files", max_steps=1)
    assert out["steps"] == 1
    assert out["success"] is True
    assert any("ls -la" in c for c in calls["executed"])


def test_agent_stops_when_no_command_found(stub_model):
    calls, replies = stub_model
    replies[:] = ["I have no idea what to do."]
    out = A.run_2b_agent("vague task", max_steps=3)
    assert out["steps"] == 1, "should stop after one unproductive step"
    assert calls["executed"] == []
    assert out["success"] is False


def test_agent_respects_max_steps(stub_model, monkeypatch):
    calls, replies = stub_model
    replies[:] = ['<function name="bash"><param name="command">pwd</param></function>']
    # finish="stop" ends the loop early by design; use "length" to exercise the
    # max_steps bound instead.
    def never_stops(messages, model="minicpm5-2b", max_tokens=512, **kw):
        calls["n"] += 1
        return {"content": replies[0], "tok_s": 50.0, "elapsed": 1.0,
                "finish": "length"}
    monkeypatch.setattr(A, "call_2b", never_stops)
    out = A.run_2b_agent("loop", max_steps=3)
    assert out["steps"] == 3
    assert calls["n"] == 3


def test_agent_does_not_execute_prose(stub_model):
    # The important safety property: an English sentence is never run.
    calls, replies = stub_model
    replies[:] = ["Here is a plan.\n```\nThe agent makes 3 calls per turn\n```"]
    out = A.run_2b_agent("vague", max_steps=2)
    assert calls["executed"] == []
    assert out["success"] is False


def test_agent_result_shape(stub_model):
    calls, replies = stub_model
    replies[:] = ['<function name="bash"><param name="command">pwd</param></function>']
    out = A.run_2b_agent("t", max_steps=1)
    assert set(out) == {"avg_tok_s", "total_time", "steps", "success",
                        "functions_created", "results"}
    assert out["avg_tok_s"] == 50.0

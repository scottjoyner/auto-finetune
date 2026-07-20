"""Tests for src.parsers (shared tool-call parsers)."""
from __future__ import annotations

from src import parsers as P


# ── RefinedToolCallV5 dialect (parse_tool_calls) ──────────────────────────────
def test_parse_tool_calls_finetune_format():
    text = ('<tool_call name="write" call_id="c1">'
            '{"filePath": "a.txt", "content": "hi"}\u276E\u276E\u276E')
    calls = P.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "write"
    assert calls[0]["args"]["content"] == "hi"


def test_parse_tool_calls_escaped_sep():
    text = ('<tool_call name="bash">{"command": "ls"}\\u276E\\u276E\\u276E')
    calls = P.parse_tool_calls(text)
    assert calls[0]["name"] == "bash"
    assert calls[0]["args"]["command"] == "ls"


def test_parse_tool_calls_base_canonical_format():
    text = ('planning...\n<tool_call>\n'
            '{"name": "bash", "arguments": {"command": "ls -la"}}\n'
            '</tool_call>')
    calls = P.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "bash"
    assert calls[0]["args"]["command"] == "ls -la"


def test_parse_tool_calls_none_on_bad_json():
    text = '<tool_call name="write" call_id="1">{not json}\u276E\u276E\u276E'
    calls = P.parse_tool_calls(text)
    assert calls[0]["name"] == "write"
    assert calls[0]["args"] is None


def test_parse_tool_calls_empty():
    assert P.parse_tool_calls("just prose, no tool calls") == []


# ── native HF function-call JSON (parse_native_tool_calls) ───────────────────
def test_parse_native_both_shapes():
    a = P.parse_native_tool_calls('x {"name": "bash", "arguments": {"command": "ls"}} y')
    assert a == [{"name": "bash", "args": {"command": "ls"}}]
    b = P.parse_native_tool_calls(
        '{"function": {"name": "write", "parameters": {"filePath": "a", "content": "x"}}}')
    assert b[0]["name"] == "write"
    assert b[0]["args"]["filePath"] == "a"


def test_parse_native_nested():
    txt = 'call: {"name": "edit", "arguments": {"filePath": "f", "oldText": "a", "newText": "x"}}'
    calls = P.parse_native_tool_calls(txt)
    assert any(c["name"] == "edit" for c in calls)
    assert calls[0]["args"]["newText"] == "x"


def test_parse_native_preserves_empty_args():
    calls = P.parse_native_tool_calls('{"name": "bash", "arguments": {}}')
    assert calls == [{"name": "bash", "args": {}}]


def test_parse_native_skips_non_call_blocks():
    txt = 'noise {"random": 1} and {"name": "bash", "arguments": {}} end'
    assert P.parse_native_tool_calls(txt) == [{"name": "bash", "args": {}}]


def test_parse_native_skips_unparseable_json():
    assert P.parse_native_tool_calls('{not valid json{{') == []
    assert P.parse_native_tool_calls('{"foo": [1,2,3]}') == []


def test_balanced_brace_spans():
    text = "a {b {c} d} e {f}"
    spans = P._balanced_brace_spans(text)
    # nested braces collapse to their outermost balanced span
    assert spans == [(2, 11), (14, 17)]
    assert [text[s:e] for s, e in spans] == ["{b {c} d}", "{f}"]

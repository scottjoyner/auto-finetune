"""Tests for the LFM3 SFT converter."""
from scripts.make_lfm3_sft import convert_example


def test_converts_tool_call_to_lfm3_dialect():
    ex = {
        "messages": [
            {"role": "assistant", "content":
             '<tool_call name="bash" call_id="c1">\n{"command": "ls", "workdir": "/tmp"}\n</tool_call>'},
        ]
    }
    conv, n = convert_example(ex)
    assert n == 1
    content = conv["messages"][0]["content"]
    assert "<tool_call>" in content
    assert "<function=bash>" in content
    assert "<parameter=command>" in content
    assert "ls" in content
    assert "<parameter=workdir>" in content
    assert "/tmp" in content
    assert "</function>" in content
    assert "</tool_call>" in content
    assert "<tool_call name=" not in content


def test_preserves_non_call_text():
    ex = {"messages": [{"role": "assistant", "content": "hello world"}]}
    conv, n = convert_example(ex)
    assert n == 0
    assert conv["messages"][0]["content"] == "hello world"


def test_aliases_terminal():
    ex = {"messages": [{"role": "assistant", "content":
                        '<tool_call name="terminal" call_id="c2">\n{"command": "pwd"}\n</tool_call>'}]}
    conv, n = convert_example(ex)
    assert n == 1
    assert "<function=bash>" in conv["messages"][0]["content"]

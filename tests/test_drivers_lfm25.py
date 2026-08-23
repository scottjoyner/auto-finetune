"""Tests for the LFM2.5 pythonic tool-call dialect parser and driver plumbing."""
import json

import pytest

from src.drivers_lfm25 import (LFM25Driver, _is_wrapped_tool_result,
                               parse_pythonic_tool_calls)


class TestParsePythonicToolCalls:
    def test_single_call_with_prose(self):
        text = ('<|tool_call_start|>[bash(command="ls -la")]<|tool_call_end|>'
                "I will list the files.")
        calls = parse_pythonic_tool_calls(text)
        assert calls == [{"name": "bash", "args": {"command": "ls -la"}}]

    def test_parallel_batched_calls(self):
        # Probed habit: independent actions batch into ONE list.
        text = ('<|tool_call_start|>[bash(command="cat a"), '
                'bash(command="cat b")]<|tool_call_end|>')
        calls = parse_pythonic_tool_calls(text)
        assert [c["name"] for c in calls] == ["bash", "bash"]
        assert calls[1]["args"] == {"command": "cat b"}

    def test_nested_quote_escaping(self):
        # Probed habit: model escapes inner quotes correctly.
        text = ('<|tool_call_start|>[write(filePath="/tmp/hello.py", '
                'content="print(\\"hi there\\")")]<|tool_call_end|>')
        calls = parse_pythonic_tool_calls(text)
        assert calls[0]["args"]["content"] == 'print("hi there")'

    def test_literal_types(self):
        text = '<|tool_call_start|>[edit(filePath="f", oldString="a", newString="b")]<|tool_call_end|>'
        calls = parse_pythonic_tool_calls(text)
        assert calls[0]["args"] == {"filePath": "f", "oldString": "a",
                                    "newString": "b"}

    def test_missing_end_token_tolerated(self):
        text = '<|tool_call_start|>[read(filePath="/etc/hostname")]'
        calls = parse_pythonic_tool_calls(text)
        assert calls[0]["name"] == "read"

    def test_no_calls(self):
        assert parse_pythonic_tool_calls("Just a plain answer.") == []

    def test_malformed_yields_recovery_entry(self):
        # Bare name instead of call -> recovery entry with args=None so the
        # bench loop's self-correction message engages.
        text = '<|tool_call_start|>[bash]<|tool_call_end|>'
        calls = parse_pythonic_tool_calls(text)
        assert calls == [{"name": "", "args": None}]

    def test_positional_args_rejected_to_recovery(self):
        text = '<|tool_call_start|>[bash("ls")]<|tool_call_end|>'
        calls = parse_pythonic_tool_calls(text)
        assert calls[0]["name"] == "bash"
        assert calls[0]["args"] is None

    def test_container_arg_value(self):
        text = ('<|tool_call_start|>[write(filePath="x", '
                'content="line1\\nline2")]<|tool_call_end|>')
        calls = parse_pythonic_tool_calls(text)
        assert calls[0]["args"]["content"] == "line1\nline2"


class TestDriverPlumbing:
    def test_wrap_result_shape_matches_trained_format(self):
        wrapped = LFM25Driver.wrap_result("some stdout")
        data = json.loads(wrapped)
        assert isinstance(data, list) and "output" in data[0]
        assert _is_wrapped_tool_result(wrapped)

    def test_translate_maps_wrapped_results_to_tool_role(self):
        msgs = [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "<|tool_call_start|>[bash(command=\"ls\")]<|tool_call_end|>"},
            {"role": "user", "content": LFM25Driver.wrap_result("file1\n")},
        ]
        out = LFM25Driver._translate(msgs, advertise_tools=True)
        assert out[0]["role"] == "system"
        assert "List of tools" in out[0]["content"]
        assert "be helpful" in out[0]["content"]
        assert out[-1]["role"] == "tool"

    def test_translate_injects_tools_doc_when_absent(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = LFM25Driver._translate(msgs, advertise_tools=True)
        assert out[0]["role"] == "system"
        assert "List of tools" in out[0]["content"]

    def test_registration(self):
        from src.bench import RUNNERS
        assert "lfm25" in RUNNERS

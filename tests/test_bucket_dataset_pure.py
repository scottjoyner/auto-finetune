"""Offline, deterministic coverage for the pure helpers in ``src.bucket_dataset``.

Only the parsing/bucketing helpers are exercised here:

* ``_tool_names`` -- tool-call extraction from message dicts, and
* ``bucket_example`` -- behavior/per-tool label assignment.

The file-I/O ``main()`` path is deliberately *not* called: it reads a real
dataset and writes bucket files plus a manifest. No GPU, network, DB or large
files are touched by this suite.
"""
from __future__ import annotations

import pytest

from src import bucket_dataset as B


def _messages(*names: str) -> list[dict]:
    """Build messages carrying legacy-style ``{"name": ...}`` tool calls."""
    return [{"tool_calls": [{"name": name} for name in names]}]


# --------------------------------------------------------------------------- #
# _tool_names (parsing)
# --------------------------------------------------------------------------- #

def test_tool_names_empty_messages():
    assert B._tool_names([]) == set()


def test_tool_names_message_without_tool_calls():
    assert B._tool_names([{"role": "user", "content": "hi"}]) == set()


def test_tool_names_null_tool_calls_is_treated_as_empty():
    assert B._tool_names([{"tool_calls": None}]) == set()


def test_tool_names_legacy_direct_name():
    messages = [{"tool_calls": [{"name": "read_file"}, {"name": "write_file"}]}]
    assert B._tool_names(messages) == {"read_file", "write_file"}


def test_tool_names_openai_function_wrapper():
    messages = [{"tool_calls": [{"function": {"name": "read_file"}}]}]
    assert B._tool_names(messages) == {"read_file"}


def test_tool_names_dedupes_across_messages():
    messages = [
        {"tool_calls": [{"name": "read_file"}, {"name": "patch"}]},
        {"tool_calls": [{"function": {"name": "read_file"}}]},
    ]
    assert B._tool_names(messages) == {"read_file", "patch"}


def test_tool_names_missing_name_defaults_to_unknown():
    assert B._tool_names([{"tool_calls": [{"arguments": "{}"}]}]) == {"unknown"}


def test_tool_names_non_dict_tool_calls_default_to_unknown():
    assert B._tool_names([{"tool_calls": ["not-a-dict", 42]}]) == {"unknown"}


def test_tool_names_non_dict_function_payload_defaults_to_unknown():
    assert B._tool_names([{"tool_calls": [{"function": "read_file"}]}]) == {"unknown"}


def test_tool_names_null_function_payload_defaults_to_unknown():
    assert B._tool_names([{"tool_calls": [{"function": None}]}]) == {"unknown"}


def test_tool_names_none_messages_raises_type_error():
    with pytest.raises(TypeError):
        B._tool_names(None)


def test_tool_names_non_dict_message_raises_attribute_error():
    with pytest.raises(AttributeError):
        B._tool_names(["not-a-message"])


# --------------------------------------------------------------------------- #
# bucket_example (bucketing)
# --------------------------------------------------------------------------- #

def test_bucket_example_missing_messages_is_general():
    assert B.bucket_example({}) == ["general"]


def test_bucket_example_empty_messages_is_general():
    assert B.bucket_example({"messages": []}) == ["general"]


def test_bucket_example_no_tool_calls_is_general():
    assert B.bucket_example({"messages": [{"content": "hi"}]}) == ["general"]


def test_bucket_example_single_known_tool_non_overlap_is_per_tool():
    assert B.bucket_example({"messages": _messages("read_file")}) == ["tool:read_file"]


def test_bucket_example_single_known_tool_overlap_adds_group():
    assert B.bucket_example({"messages": _messages("read_file")}, overlap=True) == [
        "filesystem",
        "tool:read_file",
    ]


def test_bucket_example_non_overlap_picks_alphabetically_first_tool():
    example = {"messages": _messages("write_file", "read_file", "execute_code")}
    assert B.bucket_example(example) == ["tool:execute_code"]


def test_bucket_example_overlap_sorts_group_and_tool_labels():
    example = {"messages": _messages("write_file", "read_file", "kg_query")}
    assert B.bucket_example(example, overlap=True) == [
        "filesystem",
        "knowledge_graph",
        "tool:kg_query",
        "tool:read_file",
        "tool:write_file",
    ]


def test_bucket_example_cross_group_overlap():
    example = {
        "messages": _messages("session_search", "send_message", "sophia_voice_status")
    }
    assert B.bucket_example(example, overlap=True) == [
        "messaging",
        "session",
        "tool:send_message",
        "tool:session_search",
        "tool:sophia_voice_status",
        "voice",
    ]


def test_bucket_example_unmapped_tool_is_still_a_per_tool_bucket():
    example = {"messages": _messages("totally_custom_tool")}
    assert B.bucket_example(example) == ["tool:totally_custom_tool"]
    assert B.bucket_example(example, overlap=True) == ["tool:totally_custom_tool"]


def test_bucket_example_malformed_tool_call_yields_unknown_bucket():
    example = {"messages": [{"tool_calls": [{"arguments": "{}"}]}]}
    assert B.bucket_example(example) == ["tool:unknown"]


def test_bucket_example_reads_openai_function_payload():
    example = {"messages": [{"tool_calls": [{"function": {"name": "read_file"}}]}]}
    assert B.bucket_example(example) == ["tool:read_file"]


def test_bucket_example_none_messages_raises_type_error():
    with pytest.raises(TypeError):
        B.bucket_example({"messages": None})


@pytest.mark.parametrize(
    "tool,group",
    [
        ("read_file", "filesystem"),
        ("execute_code", "code_execution"),
        ("kg_query", "knowledge_graph"),
        ("memory", "memory"),
        ("todo", "planning"),
        ("session_search", "session"),
        ("browser_navigate", "browser"),
        ("sophia_voice_status", "voice"),
        ("send_message", "messaging"),
    ],
)
def test_every_behavior_group_is_mapped(tool, group):
    example = {"messages": _messages(tool)}
    assert B.bucket_example(example, overlap=True) == sorted([group, f"tool:{tool}"])

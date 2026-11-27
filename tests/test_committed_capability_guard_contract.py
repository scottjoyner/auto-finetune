"""Offline contract for the committed capability-guard declaration.

This is a drift guard over committed files only. It reads the tracked JSON
declaration and asks git which paths are committed; it never opens a database,
contacts a network, reads credentials, starts a service or loads a model.
The declaration is a standalone artifact (no runtime consumer in the repo), so
this test guards its declared read-only posture rather than claiming runtime
enforcement.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD_REL = "filters/capability-guards.json"

READ_ONLY_TOOLS = {"read", "summarize", "flag", "report"}
WRITE_TOOLS = {"write", "execute", "modify", "delete", "deploy"}
TOP_LEVEL = {
    "max_context_tokens", "max_output_tokens", "temperature", "tool_use",
    "safety", "output_format", "filtering",
}


def _is_committed(rel: str) -> bool:
    proc = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == rel


def _guard_from_tree() -> dict:
    assert _is_committed(GUARD_REL), "guard declaration must be a committed file"
    return json.loads((ROOT / GUARD_REL).read_text(encoding="utf-8"))


def _assert_contract(data: dict) -> None:
    assert isinstance(data, dict)
    assert set(data) == TOP_LEVEL

    assert isinstance(data["max_context_tokens"], int)
    assert isinstance(data["max_output_tokens"], int)
    assert 0 < data["max_output_tokens"] < data["max_context_tokens"]
    assert isinstance(data["temperature"], (int, float))
    assert 0.0 <= data["temperature"] <= 1.0

    tools = data["tool_use"]
    assert isinstance(tools, dict)
    allowed = set(tools["allowed"])
    denied = set(tools["denied"])
    assert allowed <= READ_ONLY_TOOLS, "allowed tools must stay read-only"
    assert WRITE_TOOLS <= denied, "write/execute tools must stay denied"
    assert not (allowed & denied), "a tool cannot be both allowed and denied"
    assert tools["approval_required"] is True

    safety = data["safety"]
    assert isinstance(safety, dict)
    assert safety["escalation"] == "human_review"
    deny_list = safety["deny_list"]
    assert isinstance(deny_list, list) and deny_list
    assert all(isinstance(v, str) and v for v in deny_list)
    assert not (allowed & set(deny_list)), "safety deny list overlaps allowed tools"

    fmt = data["output_format"]
    assert fmt["machine"] == "json"
    assert fmt["human"] == "markdown"
    assert isinstance(fmt["tags"], list) and fmt["tags"]

    filtering = data["filtering"]
    assert isinstance(filtering, dict)
    for key in ("suppress", "elevate"):
        assert isinstance(filtering[key], list) and filtering[key]
        assert all(isinstance(v, str) and v for v in filtering[key])


def test_committed_guard_declaration_satisfies_read_only_contract():
    _assert_contract(_guard_from_tree())


def test_guard_declaration_is_the_committed_path():
    assert _is_committed(GUARD_REL)
    assert not _is_committed("filters/absent-guard-fixture.json")


def test_negative_controls_reject_broadened_or_malformed_declarations():
    base = _guard_from_tree()
    _assert_contract(base)  # controls start from a valid tree

    widened = copy.deepcopy(base)
    widened["tool_use"]["allowed"] = sorted(set(widened["tool_use"]["allowed"]) | {"write"})
    with pytest.raises(AssertionError):
        _assert_contract(widened)

    autoapproved = copy.deepcopy(base)
    autoapproved["tool_use"]["approval_required"] = False
    with pytest.raises(AssertionError):
        _assert_contract(autoapproved)

    overlapping = copy.deepcopy(base)
    overlapping["tool_use"]["denied"] = []
    with pytest.raises(AssertionError):
        _assert_contract(overlapping)

    missing_key = copy.deepcopy(base)
    del missing_key["safety"]
    with pytest.raises(AssertionError):
        _assert_contract(missing_key)

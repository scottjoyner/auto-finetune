"""Tests for the error-mining -> contrastive repair miner (src.contrast)."""
import json
from pathlib import Path

from src.contrast import _target_file, mine_repairs


def _rec(session_id, messages):
    return {"session_id": session_id, "source": "test", "messages": messages}


def test_target_file_variants() -> None:
    assert _target_file("write", {"filePath": "/a/b/foo.txt"}) == "foo.txt"
    assert _target_file("edit", {"path": "x/y.md"}) == "y.md"
    assert _target_file("terminal", {"command": "ls"}) is None
    assert _target_file("patch", {"patch": "--- a\n+++ b/src/z.py\n@@\n+line\n"}) == "z.py"


def test_mine_repairs_finds_self_repair(tmp_path: Path) -> None:
    # session: write foo.txt -> error result, then later edit foo.txt -> ok
    rec = _rec("s1", [
        {"role": "user", "parts": [{"type": "text", "content": "fix foo.txt"}]},
        {"role": "assistant", "parts": [
            {"type": "tool", "tool": "write",
             "input": {"filePath": "foo.txt", "content": "wrong"}}]},
        {"role": "user", "parts": [{"type": "text",
                                    "content": "error: permission denied"}]},
        {"role": "assistant", "parts": [
            {"type": "tool", "tool": "edit",
             "input": {"filePath": "foo.txt", "newText": "right"}}]},
        {"role": "user", "parts": [{"type": "text", "content": "updated foo.txt ok"}]},
    ])
    (tmp_path / "cleaned").mkdir()
    (tmp_path / "cleaned" / "test_s1.json").write_text(json.dumps(rec))
    failures = tmp_path / "failures.jsonl"
    failures.write_text(json.dumps({"session_id": "s1", "bucket": "debug"}) + "\n")
    out = tmp_path / "repairs.jsonl"

    n, tax = mine_repairs(str(tmp_path / "cleaned"), str(failures), str(out))

    assert n == 1, tax
    pr = json.loads(out.read_text().splitlines()[0])
    assert pr["rejected_call"]["name"] == "write"
    assert pr["chosen_call"]["name"] == "edit"
    assert pr["target"] == "foo.txt"
    assert pr["error_marker"] == "error:"
    assert tax["repaired"] == 1
    assert tax["by_tool"]["write"] == 1
    # prompt excludes the erroneous assistant turn (ends before it)
    assert pr["prompt_messages"][-1]["role"] == "user"


def test_mine_repairs_skips_non_target_errors(tmp_path: Path) -> None:
    # terminal failure with no file target -> no pair, counted as no_target
    rec = _rec("s2", [
        {"role": "user", "parts": [{"type": "text", "content": "run build"}]},
        {"role": "assistant", "parts": [
            {"type": "tool", "tool": "terminal", "input": {"command": "make"}}]},
        {"role": "user", "parts": [{"type": "text",
                                    "content": "make: *** error: permission denied"}]},
    ])
    (tmp_path / "cleaned").mkdir()
    (tmp_path / "cleaned" / "test_s2.json").write_text(json.dumps(rec))
    failures = tmp_path / "failures.jsonl"
    failures.write_text(json.dumps({"session_id": "s2", "bucket": "debug"}) + "\n")

    n, tax = mine_repairs(str(tmp_path / "cleaned"), str(failures),
                           str(tmp_path / "repairs.jsonl"))
    assert n == 0
    assert tax["no_target"] == 1

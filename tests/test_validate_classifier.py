"""Tests for the classifier-validation harness (src.validate_classifier)."""
import json
from pathlib import Path

from pytest import approx

from src.validate_classifier import _view, build_sheet, score


def _rec(session_id, messages):
    return {"session_id": session_id, "source": "test", "messages": messages}


def test_view_pulls_first_user_and_actions() -> None:
    rec = _rec("s1", [
        {"role": "user", "parts": [{"type": "text", "text": "fix the build"}]},
        {"role": "assistant", "parts": [
            {"type": "tool", "tool": "write_file",
             "input": {"filePath": "/a/b/foo.py", "content": "x"}},
            {"type": "tool", "tool": "terminal",
             "input": {"command": "pytest"}}]},
    ])
    first, actions = _view(rec)
    assert first == "fix the build"
    assert actions[0] == "write_file(foo.py)"
    assert actions[1].startswith("terminal:pytest")


def test_score_computes_per_bucket_and_error(tmp_path: Path) -> None:
    rows = [
        {"session_id": "a", "pred_bucket": "debug", "true_bucket": "debug",
         "pred_is_error": True, "true_is_error": True, "true_cmd_error": True,
         "pred_difficulty": "hard", "true_difficulty": "hard"},
        {"session_id": "b", "pred_bucket": "debug", "true_bucket": "reasoning",
         "pred_is_error": True, "true_is_error": False, "true_cmd_error": True,
         "pred_difficulty": "easy", "true_difficulty": "easy"},
        {"session_id": "c", "pred_bucket": "reasoning", "true_bucket": "reasoning",
         "pred_is_error": False, "true_is_error": False, "true_cmd_error": False,
         "pred_difficulty": "easy", "true_difficulty": "easy"},
    ]
    p = tmp_path / "sheet.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = score(str(p))
    assert m["n_labeled"] == 3
    assert m["bucket_accuracy"] == approx(2 / 3, abs=1e-2)
    # command-error axis: a=tp, b=tp (recovered cmd error), c=tn
    #   -> precision 1.0, recall 1.0, tn 1
    assert m["cmd_error"]["precision"] == approx(1.0)
    assert m["cmd_error"]["recall"] == approx(1.0)
    assert m["cmd_error"]["tn"] == 1
    # session-failure axis: a=tp, b=fp (recovered), c=tn
    #   -> precision 0.5, recall 1.0, tn 1
    assert m["session_failure"]["precision"] == approx(1 / 2)
    assert m["session_failure"]["recall"] == approx(1.0)
    assert m["session_failure"]["tn"] == 1
    assert m["difficulty_accuracy"] == 1.0


def test_build_sheet_stratified_sample(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    fail = _rec("f1", [
        {"role": "user", "parts": [{"type": "text", "text": "run it"}]},
        {"role": "assistant", "parts": [
            {"type": "tool", "tool": "terminal",
             "input": {"command": "make"}, "output": "error: boom"}]},
    ])
    ok = _rec("o1", [
        {"role": "user", "parts": [{"type": "text", "text": "hello"}]},
    ])
    (cleaned / "t_f1.json").write_text(json.dumps(fail))
    (cleaned / "t_o1.json").write_text(json.dumps(ok))
    (tmp_path / "failures.jsonl").write_text(
        json.dumps({"session_id": "f1", "bucket": "debug"}) + "\n")

    out = tmp_path / "sheet.jsonl"
    n = build_sheet(str(cleaned), str(tmp_path / "failures.jsonl"),
                        str(out), n_fail=1, n_ok=1, seed=1)
    assert n == 2
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    by_id = {r["session_id"]: r for r in rows}
    assert by_id["f1"]["pred_is_error"] is True
    assert by_id["o1"]["pred_is_error"] is False
    # f1 is a 1-tool errored session with no debug intent -> reasoning
    # (the old cascade forced any shell error -> debug, which over-predicted)
    assert by_id["f1"]["pred_bucket"] == "reasoning"
    assert by_id["o1"]["pred_bucket"] == "reasoning"

"""Coverage + correctness tests for src.analyze (CPU-only corpus analysis)."""
from __future__ import annotations

import json

from src import analyze as A


def _rec(source, session_id, messages):
    return {"source": source, "session_id": session_id, "messages": messages}


def _tool(name, inp=None, out=None):
    p = {"type": "tool", "tool": name}
    if inp is not None:
        p["input"] = inp
    if out is not None:
        p["output"] = out
    return p


def _text(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def test_extract_features_shell():
    rec = _rec("opencode", "s1", [
        _text("user", "run the linter"),
        {"role": "assistant", "parts": [_tool("bash", {"command": "make"}, "ok")]},
        {"role": "assistant", "parts": [_tool("bash", {"command": "make test"}, "ok")]},
        {"role": "assistant", "parts": [_tool("bash", {"command": "make lint"}, "ok")]},
    ])
    f = A.extract_features(rec)
    assert f["groups"]["shell"] == 3
    assert f["n_tool"] == 3
    assert f["intent"] == []
    assert f["has_error"] is False


def test_extract_features_error_detection():
    rec = _rec("opencode", "s2", [
        _text("user", "do x"),
        {"role": "assistant", "parts": [_tool("bash", {"command": "ls"},
                                                 "ls: cannot access 'x': No such file or directory")]},
    ])
    f = A.extract_features(rec)
    assert f["has_error"] is True
    assert "no such file" in f["error_snippet"].lower()


def test_extract_features_ext_and_intent():
    rec = _rec("hermes", "h1", [
        _text("user", "refactor this module"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/a/b.py", "content": "x=1"}, "ok")]},
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/a/c.py", "content": "y=2"}, "ok")]},
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/a/d.py", "content": "z=3"}, "ok")]},
    ])
    f = A.extract_features(rec)
    assert ".py" in f["exts"]
    assert "refactor" in f["intent"]
    assert f["distinct_files"] == 3


def test_classify_bucket_priority():
    shell = A.extract_features(_rec("opencode", "x", [
        _text("user", "run stuff"),
        {"role": "assistant", "parts": [_tool("bash", {"command": "a"}, "1")] * 3},
    ]))
    assert A.classify_bucket(shell) == "shell"

    edit1 = A.extract_features(_rec("opencode", "x", [
        _text("user", "edit file"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/a/b.py", "content": "x"}, "1")]},
    ]))
    assert A.classify_bucket(edit1) == "file-edit"

    edit3 = A.extract_features(_rec("opencode", "x", [
        _text("user", "refactor"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": f"/a/{i}.py", "content": "x"}, "1") for i in range(3)]},
    ]))
    assert A.classify_bucket(edit3) == "multi-file-refactor"

    web = A.extract_features(_rec("opencode", "x", [
        _text("user", "look it up"),
        {"role": "assistant", "parts": [_tool("web", {"url": "http://x"}, "1")]},
    ]))
    assert A.classify_bucket(web) == "web-research"

    search = A.extract_features(_rec("opencode", "x", [
        _text("user", "find usage"),
        {"role": "assistant", "parts": [_tool("grep", {"pattern": "foo"}, "1")]},
    ]))
    assert A.classify_bucket(search) == "code-search"

    data = A.extract_features(_rec("opencode", "x", [
        _text("user", "analyze this csv"),
        {"role": "assistant", "parts": [_tool("python", {"code": "1"}, "1")]},
    ]))
    assert A.classify_bucket(data) == "data-analysis"

    reasoning = A.extract_features(_rec("opencode", "x", [
        _text("user", "explain closures"),
        _text("assistant", "a closure is..."),
    ]))
    assert A.classify_bucket(reasoning) == "reasoning"

    mixed = A.extract_features(_rec("opencode", "x", [
        _text("user", "do thing"),
        {"role": "assistant", "parts": [_tool("mystery", {"x": 1}, "1"),
                                         _tool("mystery", {"x": 2}, "1")]},
    ]))
    assert A.classify_bucket(mixed) == "mixed"


def test_classify_difficulty():
    easy = A.extract_features(_rec("o", "x", [
        _text("user", "a"),
        _text("assistant", "b"),
        {"role": "assistant", "parts": [_tool("bash", {"command": "ls"}, "1")]},
    ]))
    assert A.classify_difficulty(easy) == "easy"

    hard = A.extract_features(_rec("o", "x", [
        _text("user", "a"),
    ] + [{"role": "assistant", "parts": [_tool("bash", {"command": str(i)}, "1")]}
         for i in range(15)]))
    assert A.classify_difficulty(hard) == "hard"


def test_quality_flag():
    short = A.extract_features(_rec("o", "x", [_text("user", "a"), _text("assistant", "b")]))
    keep, reason = A.quality_flag(short, {})
    assert keep is False and reason == "too_short"


def test_derive_task_from_successful_edit():
    rec = _rec("hermes", "h9", [
        _text("user", "Please create a helper that returns the sum of a list"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/repo/sum.py", "content": "def sum(xs):\n    return sum(xs)"},
                   "ok")]},
    ])
    f = A.extract_features(rec)
    meta = {"keep": True, "bucket": "file-edit", "difficulty": "easy", "source": "hermes",
            "features": f, "session_id": "h9"}
    task = A.derive_task(rec, meta)
    assert task is not None
    assert task["kind"] == "exec"
    assert task["checks"][0]["kind"] == "file_contains"
    assert task["checks"][0]["path"] == "sum.py"


def test_derive_task_skips_non_actionable():
    rec = _rec("opencode", "x", [_text("user", "explain recursion"),
                                  _text("assistant", "recursion is...")])
    f = A.extract_features(rec)
    meta = {"keep": True, "bucket": "reasoning", "difficulty": "easy",
            "source": "opencode", "features": f, "session_id": "x"}
    assert A.derive_task(rec, meta) is None


def test_compute_stats_aggregates():
    m = []
    for i in range(3):
        f = A.extract_features(_rec("opencode", f"o{i}", [
            _text("user", "run"),
            {"role": "assistant", "parts": [_tool("bash", {"command": str(i)}, "1")]}]))
        m.append({"source": "opencode", "bucket": "shell", "difficulty": "easy",
                  "features": f, "hash": f"h{i}"})
    # add a hermes duplicate-hash entry to exercise overlap bookkeeping
    f2 = A.extract_features(_rec("hermes", "h", [_text("user", "run"),
                              {"role": "assistant", "parts": [_tool("bash", {"command": "0"}, "1")]}]))
    m.append({"source": "hermes", "bucket": "shell", "difficulty": "easy",
              "features": f2, "hash": "h0"})
    stats = A.compute_stats(m)
    assert stats["total"] == 4
    assert stats["by_source"]["opencode"] == 3
    assert stats["by_bucket"]["shell"] == 4
    assert stats["top_tools"][0][0] == "bash"


def test_analyze_all_writes_artifacts(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    # shell session
    (cleaned / "a.json").write_text(json.dumps(_rec("opencode", "a", [
        _text("user", "run the tests"),
        {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}])))
    # successful file-edit -> becomes a task
    (cleaned / "b.json").write_text(json.dumps(_rec("hermes", "b", [
        _text("user", "Please create a module that adds two numbers together safely"),
        {"role": "assistant", "parts": [
            _tool("write", {"filePath": "/repo/add.py", "content": "def add(a,b):\n    return a+b"},
                   "ok")]},
        _text("assistant", "Done — added the module.")])))
    # failure session -> goes to failures.jsonl
    (cleaned / "c.json").write_text(json.dumps(_rec("opencode", "c", [
        _text("user", "fix it"),
        {"role": "assistant", "parts": [
            _tool("bash", {"command": "ls"}, "ls: cannot access 'x': No such file or directory")]}])))

    out = tmp_path / "analysis"
    summary = A.analyze_all(str(cleaned), out_dir=str(out))
    assert summary["n_sessions"] == 3
    assert summary["n_tasks"] >= 1
    assert summary["n_failures"] == 1
    assert (out / "buckets.json").exists()
    assert (out / "corpus.json").exists()
    assert (out / "auto-tasks.jsonl").exists()
    assert (out / "failures.jsonl").exists()
    tasks = [json.loads(l) for l in (out / "auto-tasks.jsonl").read_text().splitlines()]
    assert any(t["task_id"] == "auto-hermes-b" for t in tasks)


def test_analyze_dedups_by_session(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    short = [_text("user", "do x"),
             {"role": "assistant", "parts": [_tool("bash", {"command": "ls"}, "ok")]}]
    longer = short + [{"role": "assistant", "parts": [_tool("bash", {"command": "pwd"}, "ok")]}]
    # same session_id in two files (cross-source snapshot) -> counted once
    (cleaned / "a1.json").write_text(json.dumps(_rec("opencode", "dup", short)))
    (cleaned / "a2.json").write_text(json.dumps(_rec("opencode", "dup", longer)))
    out = tmp_path / "analysis"
    summary = A.analyze_all(str(cleaned), out_dir=str(out))
    assert summary["n_sessions"] == 1
    b = json.loads((out / "buckets.json").read_text())
    assert set(b) == {"dup"}


def test_benchmark_session_ids(tmp_path):
    tasks = tmp_path / "auto-tasks.jsonl"
    tasks.write_text("\n".join([
        json.dumps({"task_id": "auto-hermes-20260529_205350_e6f596", "source": "hermes"}),
        json.dumps({"task_id": "auto-opencode-ses_0fb0c0944ffeuDsob6", "source": "opencode"}),
        json.dumps({"task_id": "manual-foo", "source": "hermes"}),  # not an auto-task
    ]) + "\n")
    ids = A.benchmark_session_ids(tasks)
    assert ids == {"20260529_205350_e6f596", "ses_0fb0c0944ffeuDsob6"}


def test_benchmark_session_ids_missing_file():
    assert A.benchmark_session_ids("/no/such/file.jsonl") == set()

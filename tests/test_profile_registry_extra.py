"""Offline edge/failure coverage for src.profile and src.registry pure helpers.

Complements the happy-path tests in test_cpu_heavy.py and test_autoharvest.py.
No network, GPU, DB, model downloads or heavy filesystem writes are used.
"""
from __future__ import annotations

import json

import pytest

from conftest import make_cfg


# ---------------------------------------------------------------------------
# src.profile helpers
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty_and_scaling():
    from src.profile import _estimate_tokens

    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a b c d e") == 6          # int(5 * 1.3)
    assert _estimate_tokens("one two three four five six seven eight nine ten") == 13
    assert _estimate_tokens("a b") < _estimate_tokens("a b c d e f g h")


def test_code_complexity_counts_and_nesting():
    from src.profile import _code_complexity

    text = (
        "# comment\n"
        "\n"
        "if x:\n"
        "    for i in y:\n"
        "        z = a and b or c\n"
        "def f():\n"
        "    return z"
    )
    assert _code_complexity(text) == {
        "n_lines": 7,
        "n_code_lines": 5,
        "n_blank_lines": 1,
        "n_comment_lines": 1,
        "cyclomatic_complexity": 5,   # if + for + and + or + 1
        "max_nesting_depth": 3,
    }


def test_code_complexity_empty_text():
    from src.profile import _code_complexity

    result = _code_complexity("")
    assert result == {
        "n_lines": 1,
        "n_code_lines": 0,
        "n_blank_lines": 1,
        "n_comment_lines": 0,
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 0,
    }


def test_histogram_empty_single_value_and_upper_clamp():
    from src.profile import _histogram

    assert _histogram([]) == {}
    assert _histogram([5, 5, 5]) == {"5": 3}
    assert _histogram([0, 10], bins=2) == {"0-5": 1, "5-10": 1}
    # Values beyond the last bin are clamped into the final bucket.
    hist = _histogram([0, 1, 2, 3], bins=2)
    assert sum(hist.values()) == 4


def test_language_detect_all_branches_and_precedence():
    from src.profile import _language_detect

    assert _language_detect("def foo(): pass") == "python"
    assert _language_detect("import os") == "python"
    assert _language_detect("from x import y") == "python"
    assert _language_detect("function bar() {}") == "javascript"
    assert _language_detect("const x = 1") == "javascript"
    assert _language_detect("func main() {}") == "go"
    assert _language_detect("package main") == "go"
    assert _language_detect("fn main() {}") == "rust"
    assert _language_detect("let mut x = 1") == "rust"
    assert _language_detect("public class Foo {}") == "java"
    assert _language_detect("@Override") == "java"
    assert _language_detect("#include <stdio.h>") == "c"
    assert _language_detect("void main() {}") == "c"
    assert _language_detect("<html><body></body></html>") == "html"
    assert _language_detect("SELECT * FROM t") == "sql"
    assert _language_detect("```python") == "markdown"
    assert _language_detect("just plain prose") == "text"
    # Python patterns are checked before JavaScript patterns.
    assert _language_detect("const x = 1\ndef foo(): pass") == "python"


def test_file_type_stats_skips_non_dicts_and_folds_extensions():
    from src.profile import _file_type_stats

    sessions = [{
        "messages": [{
            "parts": [
                {"input": {"filePath": "A.PY", "path": "b.TXT",
                           "filename": 123, "file": None}},
                {"input": ["not", "a", "dict"]},
                {"type": "text", "text": "def foo(): pass"},
                {"type": "tool", "input": "not a dict", "text": 123},
            ]
        }]
    }]
    stats = _file_type_stats(sessions)
    assert stats["extensions"] == {".py": 1, ".txt": 1}
    assert stats["languages"] == {"python": 1}
    assert _file_type_stats([]) == {"extensions": {}, "languages": {}}


def test_session_length_stats_empty_and_mixed_part_types():
    from src.profile import _session_length_stats

    assert _session_length_stats([]) == {}

    session = {"messages": [{
        "parts": [
            {"type": "text", "text": "one two"},                    # 2
            {"type": "tool", "input": {"cmd": "a b c"}, "output": "x y"},  # 3 + 2
            {"type": "tool", "input": {"n": 5, "s": "z"}, "output": 123},  # 1
        ]
    }]}
    stats = _session_length_stats([session])
    assert stats["n_sessions"] == 1
    assert stats["total_tokens"] == 8
    assert stats["avg_tokens"] == 8
    assert stats["median_tokens"] == stats["min_tokens"] == stats["max_tokens"] == 8
    assert stats["p90_tokens"] == stats["p95_tokens"] == 8
    assert stats["avg_messages"] == 1.0
    assert stats["avg_tools"] == 2.0
    assert stats["token_histogram"] == {"8": 1}


def test_topic_clusters_empty_and_other_bucket():
    from src.profile import _topic_clusters

    assert _topic_clusters([]) == {
        "n_clusters": 0, "top_keywords": {}, "cluster_sizes": {}, "cluster_examples": {},
    }

    sessions = [
        {"session_id": "s1", "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "the and but"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "banana banana"}]},
        ]},
        {"session_id": "s2", "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "rocket"}]},
        ]},
    ]
    clusters = _topic_clusters(sessions, max_clusters=5)
    # Keywords are collapsed into a per-session set, so repeats count once,
    # and stop words / non-user messages are ignored.
    assert clusters["top_keywords"] == {"rocket": 1}
    assert clusters["cluster_sizes"] == {"other": 1, "rocket": 1}
    assert clusters["cluster_examples"] == {"other": ["s1"], "rocket": ["s2"]}
    assert clusters["n_clusters"] == 2


def test_profile_sessions_empty_input():
    from src.profile import profile_sessions

    profile = profile_sessions([])
    assert profile["length_stats"] == {}
    assert profile["file_stats"] == {"extensions": {}, "languages": {}}
    assert profile["topics"]["n_clusters"] == 0


# ---------------------------------------------------------------------------
# src.registry helpers not covered by test_autoharvest.py
# ---------------------------------------------------------------------------

def _registry(tmp_path):
    from src.registry import ModelRegistry

    return ModelRegistry(str(tmp_path / "registry"))


def test_registry_update_valid_unknown_and_missing(tmp_path):
    reg = _registry(tmp_path)
    entry = reg.register("combined", "/ckpt", base_model="base")

    reg.update(entry.model_id, status="deployed", eval_loss=0.25, bogus="ignored")
    updated = reg.get(entry.model_id)
    assert updated.status == "deployed"
    assert updated.eval_loss == 0.25
    assert not hasattr(updated, "bogus")

    with pytest.raises(KeyError, match="model missing not found"):
        reg.update("missing", status="archived")


def test_registry_get_active_set_active_and_reload(tmp_path):
    reg = _registry(tmp_path)
    assert reg.get_active() is None

    a = reg.register("a", "/a")
    b = reg.register("b", "/b")
    reg.set_active(a.model_id)
    assert reg.get_active().model_id == a.model_id

    # Unknown id deactivates existing active entries and saves nothing new.
    reg.set_active("missing")
    assert reg.get_active() is None

    reg.set_active(b.model_id)
    assert reg.get_active().model_id == b.model_id
    assert reg.get(a.model_id).status == "deployed"
    assert reg.get(b.model_id).status == "active"

    reopened = _registry(tmp_path)
    assert reopened.get_active().model_id == b.model_id


def test_registry_load_existing_and_empty(tmp_path):
    reg = _registry(tmp_path)
    assert reg.models == {}

    registry_path = tmp_path / "registry" / "model-registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({
        "m1": {
            "model_id": "m1", "label": "legacy", "version": 1,
            "checkpoint_path": "/legacy", "created_at": 1.0,
        }
    }))
    loaded = _registry(tmp_path)
    entry = loaded.get("m1")
    assert entry.model_id == "m1"
    assert entry.status == "trained"          # dataclass default
    assert entry.train_loss is None
    assert loaded.get("missing") is None


def test_registry_get_latest_none_and_list_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("src.registry.time.time", lambda: 100.0)
    reg = _registry(tmp_path)
    assert reg.get_latest("none") is None
    assert reg.get_active() is None
    assert reg.get_lineage("missing") == []
    assert reg.list_models() == []

    reg.register("x", "/x1")
    reg.register("x", "/x2")
    reg.register("y", "/y1")
    assert reg.get_latest("x").version == 2
    assert [m.version for m in reg.list_models("x")] == [2, 1]
    assert {m.label for m in reg.list_models()} == {"x", "y"}


def test_registry_lineage_stops_at_missing_parent(tmp_path):
    reg = _registry(tmp_path)
    a = reg.register("a", "/a")
    orphan = reg.register("orphan", "/o", parent_model="does-not-exist")
    assert [m.model_id for m in reg.get_lineage(a.model_id)] == [a.model_id]
    assert [m.model_id for m in reg.get_lineage(orphan.model_id)] == [orphan.model_id]


def test_registry_prune_protects_active_and_deployed(tmp_path, monkeypatch):
    monkeypatch.setattr("src.registry.time.time", lambda: 1.0)
    reg = _registry(tmp_path)
    entries = [reg.register("t", f"/m{i}") for i in range(5)]
    reg.models[entries[0].model_id].status = "active"
    reg.models[entries[1].model_id].status = "deployed"

    reg.prune_old_versions("t", keep=2)
    statuses = {e.version: reg.get(e.model_id).status for e in entries}
    assert statuses == {1: "active", 2: "deployed", 3: "archived", 4: "trained", 5: "trained"}

    # No candidates / keep larger than population are no-ops.
    reg.prune_old_versions("absent", keep=1)
    reg.prune_old_versions("t", keep=10)
    assert reg.get(entries[2].model_id).status == "archived"


def test_registry_main_cli_paths(tmp_path, capsys):
    cfg = make_cfg(paths={"analysis_dir": str(tmp_path / "analysis")})
    from src.registry import ModelRegistry, main

    assert main(cfg, ["prog", "registry-list"]) == 0
    assert "no models registered" in capsys.readouterr().out

    assert main(cfg, ["prog", "registry-add"]) == 2
    assert "requires --label" in capsys.readouterr().out

    assert main(cfg, ["prog", "registry-add", "--label=x", "--checkpoint=/ckpt",
                      "--base=base-model"]) == 0
    assert "registered toolcall-v5-3b-x-v1" in capsys.readouterr().out

    registry_dir = tmp_path / "analysis" / "registry"
    reg = ModelRegistry(str(registry_dir))
    entry = reg.get("toolcall-v5-3b-x-v1")
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "weights.bin").write_bytes(b"tiny")
    reg.update(entry.model_id, checkpoint_path=str(ckpt_dir),
               eval_loss=0.1234, tool_exact_match=0.875)

    assert main(cfg, ["prog", "registry-list", "--label=x"]) == 0
    out = capsys.readouterr().out
    assert "1 models:" in out
    assert "loss=0.1234" in out
    assert "tool=0.875" in out
    assert "0MB" in out

    assert main(cfg, ["prog", "bogus-command"]) == 0
    assert "Commands:" in capsys.readouterr().out

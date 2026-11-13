"""Synthetic training-bucket sampling and output-custody tests."""
from __future__ import annotations

import json
import random

import pytest

from src import auto_balance as bal


class Config:
    def __init__(self, root):
        self.root = root
    def path(self, name):
        assert name in ("analysis_dir", "cleaned_dir")
        return str(self.root / name)


def setup_case(tmp_path):
    analysis = tmp_path / "analysis_dir"
    cleaned = tmp_path / "cleaned_dir"
    analysis.mkdir()
    cleaned.mkdir()
    return Config(tmp_path), analysis, cleaned


def bucket_map():
    return {
        "s1": {"bucket": "file-edit", "source": "opencode", "keep": True, "difficulty": "easy"},
        "s2": {"bucket": "file-edit", "source": "hermes", "keep": True, "difficulty": "hard"},
        "s3": {"bucket": "debug", "source": "hermes", "keep": True},
        "s4": {"bucket": "debug", "source": "opencode", "keep": False},
    }


def test_bucket_statistics_include_filtered_source_and_difficulty():
    stats = bal.compute_balance_stats(bucket_map())
    assert stats["total"] == 4
    assert stats["kept"] == 3
    assert stats["dropped"] == 1
    assert stats["by_bucket"] == {"file-edit": 2, "debug": 2}
    assert stats["by_source"]["hermes"] == 2
    assert stats["by_difficulty"]["medium"] == 2


def test_empty_bucket_analysis_safe():
    assert bal.compute_balance_stats({})["total"] == 0
    assert bal.balance_buckets({}, target_size=10) == {}


def test_balancing_excludes_dropped_sessions():
    b = bal.balance_buckets(bucket_map(), target_size=100)
    assert set(b) == {"file-edit", "debug"}
    assert set(b["file-edit"]) == {"s1", "s2"}
    assert b["debug"] == ["s3"]
    assert "s4" not in str(b)


def test_balancing_is_deterministic_and_does_not_change_global_rng():
    data = {str(i): {"bucket": "file-edit"} for i in range(25)}
    before = random.getstate()
    first = bal.balance_buckets(data, target_size=10, seed=33)
    after = random.getstate()
    second = bal.balance_buckets(data, target_size=10, seed=33)
    assert first == second
    assert before == after
    assert len(first["file-edit"]) == 10


@pytest.mark.parametrize("target", [0, -1, 1.5, "many", None])
def test_invalid_balance_target_hold(target):
    with pytest.raises(ValueError, match="invalid_balance_target"):
        bal.balance_buckets(bucket_map(), target_size=target)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_bucket_weights_hold(weight):
    with pytest.raises(ValueError, match="invalid_bucket_weights"):
        bal.balance_buckets(bucket_map(), target_size=100,
                            weights={"file-edit": weight, "debug": 1})


def test_bucket_map_loaded_from_temp_file(tmp_path):
    path = tmp_path / "buckets.json"
    path.write_text(json.dumps(bucket_map()))
    assert bal.load_bucket_map(str(path)) == bucket_map()


def test_session_loader_ignores_corrupt_and_missing_ids(tmp_path):
    root = tmp_path / "cleaned"
    nested = root / "hermes"
    nested.mkdir(parents=True)
    (nested / "a.json").write_text(json.dumps({"session_id": "s1", "messages": []}))
    (nested / "bad.json").write_text("{bad")
    (nested / "no-id.json").write_text(json.dumps({"messages": []}))
    (nested / "nonobject.json").write_text("[]")
    loaded = bal.load_sessions(str(root))
    assert list(loaded) == ["s1"]
    assert loaded["s1"]["messages"] == []


def test_write_balanced_dataset_counts_groups_with_temp_files(tmp_path):
    target = tmp_path / "out" / "balanced.jsonl"
    selected = {"file-edit": ["s1"], "debug": ["s2"]}
    sessions = {
        "s1": {"session_id": "s1", "messages": []},
        "s2": {"session_id": "s2", "messages": []},
    }
    stats = bal.write_balanced_dataset(selected, sessions, str(target))
    assert stats["total"] == 2
    assert stats["by_bucket"] == {"debug": 1, "file-edit": 1}
    assert {json.loads(x)["session_id"] for x in target.read_text().splitlines()} == {"s1", "s2"}


def test_missing_selected_session_denies_before_creating_output(tmp_path):
    target = tmp_path / "new" / "balanced.jsonl"
    with pytest.raises(ValueError, match="selected_session_records_unavailable"):
        bal.write_balanced_dataset({"file-edit": ["missing"]}, {}, str(target))
    assert not target.exists()
    assert not target.parent.exists()


def test_cli_missing_map_returns_failure_without_output(tmp_path):
    cfg, analysis, cleaned = setup_case(tmp_path)
    assert bal.main(cfg, out_dir=str(tmp_path / "output")) == 1
    assert not (tmp_path / "output").exists()


def test_cli_complete_samples_writes_dataset_and_metadata(tmp_path):
    cfg, analysis, cleaned = setup_case(tmp_path)
    (analysis / "buckets.json").write_text(json.dumps(bucket_map()))
    for name in ("s1", "s2", "s3"):
        (cleaned / f"{name}.json").write_text(json.dumps({
            "session_id": name, "messages": [{"role": "user", "content": "synthetic"}],
        }))
    output = tmp_path / "balanced"
    assert bal.main(cfg, cap=100, out_dir=str(output), seed=42) == 0
    rows = [json.loads(s) for s in (output / "train.balanced.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert {s["session_id"] for s in rows} == {"s1", "s2", "s3"}
    meta = json.loads((output / "balance-meta.json").read_text())
    assert meta["balanced_stats"]["total"] == 3


def test_cli_missing_selected_records_hold_before_output(tmp_path):
    cfg, analysis, cleaned = setup_case(tmp_path)
    (analysis / "buckets.json").write_text(json.dumps(bucket_map()))
    (cleaned / "s1.json").write_text(json.dumps({"session_id": "s1"}))
    target = tmp_path / "balanced"
    assert bal.main(cfg, cap=100, out_dir=str(target)) == 78
    assert not target.exists()


def test_cli_zero_samples_hold_without_output(tmp_path):
    cfg, analysis, _ = setup_case(tmp_path)
    (analysis / "buckets.json").write_text(json.dumps({
        "none": {"bucket": "debug", "keep": False}
    }))
    target = tmp_path / "balanced"
    assert bal.main(cfg, cap=100, out_dir=str(target)) == 78
    assert not target.exists()


def test_cli_invalid_cap_holds_without_output(tmp_path):
    cfg, analysis, _ = setup_case(tmp_path)
    (analysis / "buckets.json").write_text(json.dumps(bucket_map()))
    target = tmp_path / "balanced"
    assert bal.main(cfg, cap=0, out_dir=str(target)) == 78
    assert not target.exists()

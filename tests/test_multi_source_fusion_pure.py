"""Focused offline tests for the deterministic surface of src.multi_source_fusion.

All functions exercised here operate only on local files and in-memory data:
no network, GPU, database, or model downloads are involved. Timing fields are
asserted only for non-negativity (they are inherently non-deterministic).
"""
from __future__ import annotations

import json

import pytest

import src.multi_source_fusion as mmsf
from src.multi_source_fusion import (
    FusionStats,
    fuse_sources,
    freshness_weighted_fuse,
    load_source,
)


class _FakeCfg:
    """Offline config stub: only the dataset_dir lookup the CLI needs."""

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def path(self, name):
        assert name == "dataset_dir"
        return self.dataset_dir


def _ex(name, messages=None):
    """Two-message example by default (passes the quality check)."""
    if messages is None:
        messages = [
            {"role": "user", "content": f"q-{name}"},
            {"role": "assistant", "content": f"a-{name}"},
        ]
    return {"id": name, "messages": messages}


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# FusionStats
# --------------------------------------------------------------------------

def test_fusion_stats_defaults():
    s = FusionStats()
    assert s.sources == []
    assert s.input_counts == {}
    assert s.output_count == 0
    assert s.duplicates_removed == 0
    assert s.quality_filtered == 0
    assert s.fusion_time == 0.0


def test_fusion_stats_mutable_defaults_are_independent():
    first = FusionStats()
    first.sources.append("opencode")
    first.input_counts["opencode"] = 3
    second = FusionStats()
    assert second.sources == []
    assert second.input_counts == {}


def test_fusion_stats_accepts_explicit_values():
    s = FusionStats(sources=["a"], input_counts={"a": 1}, output_count=1,
                    duplicates_removed=2, quality_filtered=3, fusion_time=0.5)
    assert s.sources == ["a"]
    assert s.input_counts == {"a": 1}
    assert (s.output_count, s.duplicates_removed, s.quality_filtered) == (1, 2, 3)
    assert s.fusion_time == 0.5
    assert set(json.loads(json.dumps(s.__dict__))) == {
        "sources", "input_counts", "output_count", "duplicates_removed",
        "quality_filtered", "fusion_time",
    }


# --------------------------------------------------------------------------
# load_source
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    "train.opencode.jsonl",
    "train_opencode.jsonl",
    "opencode.jsonl",
])
def test_load_source_recognized_patterns(tmp_path, filename):
    records = [_ex("one"), _ex("two")]
    _write_jsonl(tmp_path / filename, records)
    assert load_source("opencode", str(tmp_path)) == records


def test_load_source_first_pattern_wins(tmp_path):
    _write_jsonl(tmp_path / "train.opencode.jsonl", [_ex("first")])
    _write_jsonl(tmp_path / "train_opencode.jsonl", [_ex("second")])
    _write_jsonl(tmp_path / "opencode.jsonl", [_ex("third")])
    assert load_source("opencode", str(tmp_path)) == [_ex("first")]


def test_load_source_stops_at_existing_but_empty_file(tmp_path):
    (tmp_path / "train.opencode.jsonl").write_text("")
    _write_jsonl(tmp_path / "opencode.jsonl", [_ex("fallback")])
    assert load_source("opencode", str(tmp_path)) == []


def test_load_source_missing_source_returns_empty(tmp_path):
    assert load_source("hermes", str(tmp_path)) == []


def test_load_source_skips_blank_and_malformed_lines(tmp_path):
    good = _ex("good")
    (tmp_path / "train.hermes.jsonl").write_text(
        "\n" + json.dumps(good) + "\n" + "   \n" + "{not json}\n\n"
        + json.dumps([1, 2, 3]) + "\n"
    )
    assert load_source("hermes", str(tmp_path)) == [good, [1, 2, 3]]


def test_load_source_does_not_mix_other_sources(tmp_path):
    _write_jsonl(tmp_path / "train.neo4j.jsonl", [_ex("other")])
    _write_jsonl(tmp_path / "train.opencode.jsonl", [_ex("mine")])
    assert load_source("opencode", str(tmp_path)) == [_ex("mine")]


# --------------------------------------------------------------------------
# fuse_sources
# --------------------------------------------------------------------------

def test_fuse_sources_basic_two_sources(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1"), _ex("a2")])
    _write_jsonl(tmp_path / "train.b.jsonl", [_ex("b1")])
    out = tmp_path / "out" / "fused.jsonl"
    stats = fuse_sources(["a", "b"], str(tmp_path), str(out))

    assert stats.sources == ["a", "b"]
    assert stats.input_counts == {"a": 2, "b": 1}
    assert stats.output_count == 3
    assert stats.duplicates_removed == 0
    assert stats.quality_filtered == 0
    assert stats.fusion_time >= 0.0
    assert [r["id"] for r in _read_jsonl(out)] == ["a1", "a2", "b1"]


def test_fuse_sources_dedup_across_and_within_sources(tmp_path):
    shared = _ex("shared")
    _write_jsonl(tmp_path / "train.a.jsonl", [shared, _ex("a1"), shared])
    _write_jsonl(tmp_path / "train.b.jsonl", [shared, _ex("b1")])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a", "b"], str(tmp_path), str(out), dedup=True)

    assert stats.input_counts == {"a": 3, "b": 2}
    assert stats.duplicates_removed == 2
    assert stats.output_count == 3
    assert [r["id"] for r in _read_jsonl(out)] == ["shared", "a1", "b1"]


def test_fuse_sources_dedup_uses_sorted_keys(tmp_path):
    first = {"id": "x", "messages": [], "nested": {"a": 1, "b": 2}}
    reordered = {"nested": {"b": 2, "a": 1}, "messages": [], "id": "x"}
    _write_jsonl(tmp_path / "train.a.jsonl", [first])
    _write_jsonl(tmp_path / "train.b.jsonl", [reordered])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a", "b"], str(tmp_path), str(out))
    assert stats.duplicates_removed == 1
    assert stats.output_count == 1


def test_fuse_sources_without_dedup_keeps_all(tmp_path):
    shared = _ex("shared")
    _write_jsonl(tmp_path / "train.a.jsonl", [shared, shared])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a"], str(tmp_path), str(out), dedup=False)
    assert stats.duplicates_removed == 0
    assert stats.output_count == 2
    assert _read_jsonl(out) == [shared, shared]


def test_fuse_sources_quality_filter_removes_short_records(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [
        _ex("no_messages", messages=[]),
        _ex("one_message", messages=[{"role": "user", "content": "hi"}]),
        _ex("two_messages"),
        {"id": "no_key"},
    ])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a"], str(tmp_path), str(out), min_quality=0.5)
    assert stats.quality_filtered == 3
    assert stats.output_count == 1
    assert [r["id"] for r in _read_jsonl(out)] == ["two_messages"]


def test_fuse_sources_quality_filter_disabled_at_zero(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("short", messages=[])])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a"], str(tmp_path), str(out), min_quality=0.0)
    assert stats.quality_filtered == 0
    assert stats.output_count == 1


def test_fuse_sources_quality_threshold_only_checks_presence(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [
        _ex("good"), _ex("short", messages=[{"role": "user", "content": "x"}]),
    ])
    out = tmp_path / "fused.jsonl"
    tiny = fuse_sources(["a"], str(tmp_path), str(out), min_quality=0.0001)
    high = fuse_sources(["a"], str(tmp_path), str(out), min_quality=100.0)
    assert (tiny.quality_filtered, tiny.output_count) == (1, 1)
    assert (high.quality_filtered, high.output_count) == (1, 1)


def test_fuse_sources_dedup_precedes_quality_filter(tmp_path):
    short = _ex("short", messages=[])
    _write_jsonl(tmp_path / "train.a.jsonl", [short])
    _write_jsonl(tmp_path / "train.b.jsonl", [short, _ex("good")])
    out = tmp_path / "fused.jsonl"
    stats = fuse_sources(["a", "b"], str(tmp_path), str(out), min_quality=1.0)
    assert stats.duplicates_removed == 1
    assert stats.quality_filtered == 1
    assert stats.output_count == 1
    assert [r["id"] for r in _read_jsonl(out)] == ["good"]


def test_fuse_sources_creates_nested_output_directory(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("x")])
    out = tmp_path / "deep" / "nested" / "fused.jsonl"
    fuse_sources(["a"], str(tmp_path), str(out))
    assert out.is_file()


def test_fuse_sources_relative_output_path_uses_cwd(tmp_path, monkeypatch):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("x")])
    monkeypatch.chdir(tmp_path)
    stats = fuse_sources(["a"], str(tmp_path), "relative.jsonl")
    assert stats.output_count == 1
    assert _read_jsonl(tmp_path / "relative.jsonl") == [_ex("x")]


def test_fuse_sources_empty_sources_writes_empty_file(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = fuse_sources([], str(tmp_path), str(out))
    assert stats.sources == []
    assert stats.input_counts == {}
    assert stats.output_count == 0
    assert stats.duplicates_removed == 0
    assert stats.quality_filtered == 0
    assert out.read_text() == ""


def test_fuse_sources_non_dict_record_fails_quality_filter(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [[1, 2, 3]])
    out = tmp_path / "out.jsonl"
    with pytest.raises(AttributeError):
        fuse_sources(["a"], str(tmp_path), str(out), min_quality=1.0)


def test_fuse_sources_output_parent_is_file_raises(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("x")])
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
        fuse_sources(["a"], str(tmp_path), str(blocker / "out.jsonl"))


# --------------------------------------------------------------------------
# freshness_weighted_fuse
# --------------------------------------------------------------------------

def test_weighted_fuse_default_weights_copy_once(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1"), _ex("a2")])
    _write_jsonl(tmp_path / "train.b.jsonl", [_ex("b1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(["a", "b"], str(tmp_path), str(out))

    assert stats.sources == ["a", "b"]
    assert stats.input_counts == {"a": 2, "b": 1}
    assert stats.output_count == 3
    assert stats.duplicates_removed == 0
    assert stats.quality_filtered == 0
    assert stats.fusion_time >= 0.0
    assert [r["id"] for r in _read_jsonl(out)] == ["a1", "a2", "b1"]


def test_weighted_fuse_repeats_by_integer_weight(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1"), _ex("a2")])
    _write_jsonl(tmp_path / "train.b.jsonl", [_ex("b1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a", "b"], str(tmp_path), str(out), weights={"a": 3.0, "b": 2.0}
    )
    assert stats.output_count == 8
    assert [r["id"] for r in _read_jsonl(out)] == [
        "a1", "a1", "a1", "a2", "a2", "a2", "b1", "b1",
    ]


def test_weighted_fuse_weight_below_one_still_copies_once(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a"], str(tmp_path), str(out), weights={"a": 0.4}
    )
    assert stats.output_count == 1
    assert _read_jsonl(out) == [_ex("a1")]


@pytest.mark.parametrize("weight", [0, -1, 0.9, 1.0])
def test_weighted_fuse_minimum_one_copy(tmp_path, weight):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a"], str(tmp_path), str(out), weights={"a": weight}
    )
    assert stats.output_count == 1


def test_weighted_fuse_truncates_fractional_weight(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a"], str(tmp_path), str(out), weights={"a": 2.99}
    )
    assert stats.output_count == 2


def test_weighted_fuse_missing_weight_defaults_to_one(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1")])
    _write_jsonl(tmp_path / "train.b.jsonl", [_ex("b1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a", "b"], str(tmp_path), str(out), weights={"b": 4.0}
    )
    assert stats.output_count == 5
    assert [r["id"] for r in _read_jsonl(out)] == ["a1", "b1", "b1", "b1", "b1"]


def test_weighted_fuse_dedups_across_sources_before_weighting(tmp_path):
    shared = _ex("shared")
    _write_jsonl(tmp_path / "train.a.jsonl", [shared])
    _write_jsonl(tmp_path / "train.b.jsonl", [shared, _ex("b1")])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a", "b"], str(tmp_path), str(out), weights={"a": 3.0, "b": 2.0}
    )
    assert stats.duplicates_removed == 1
    assert stats.output_count == 5
    assert [r["id"] for r in _read_jsonl(out)] == [
        "shared", "shared", "shared", "b1", "b1",
    ]


def test_weighted_fuse_does_not_apply_quality_filter(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [
        _ex("empty", messages=[]), _ex("one", messages=[{"role": "user"}]),
    ])
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse(
        ["a"], str(tmp_path), str(out), weights={"a": 2.0}
    )
    assert stats.quality_filtered == 0
    assert stats.output_count == 4
    assert [r["id"] for r in _read_jsonl(out)] == [
        "empty", "empty", "one", "one",
    ]


def test_weighted_fuse_empty_sources(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = freshness_weighted_fuse([], str(tmp_path), str(out))
    assert stats.sources == []
    assert stats.input_counts == {}
    assert stats.output_count == 0
    assert out.read_text() == ""


def test_weighted_fuse_creates_output_directory(tmp_path):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("x")])
    out = tmp_path / "nested" / "out.jsonl"
    freshness_weighted_fuse(["a"], str(tmp_path), str(out))
    assert out.is_file()


# --------------------------------------------------------------------------
# main (CLI glue; config stubbed so it stays fully offline/deterministic)
# --------------------------------------------------------------------------

def test_main_unweighted_path(tmp_path, monkeypatch, capsys):
    shared = _ex("shared")
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1"), shared, shared])
    monkeypatch.setattr(mmsf, "load", lambda: _FakeCfg(str(tmp_path)))
    out = tmp_path / "cli-out.jsonl"

    rc = mmsf.main(["--sources", "a , b", "--output", str(out),
                    "--min-quality", "1.0"])

    assert rc == 0
    assert [r["id"] for r in _read_jsonl(out)] == ["a1", "shared"]
    printed = capsys.readouterr().out
    assert "[fusion] Sources: ['a', 'b']" in printed
    assert "[fusion] Input counts: {'a': 3, 'b': 0}" in printed
    assert "[fusion] Duplicates removed: 1" in printed
    assert "[fusion] Output: 2 examples" in printed


def test_main_weighted_path(tmp_path, monkeypatch, capsys):
    _write_jsonl(tmp_path / "train.a.jsonl", [_ex("a1")])
    _write_jsonl(tmp_path / "train.b.jsonl", [_ex("b1")])
    monkeypatch.setattr(mmsf, "load", lambda: _FakeCfg(str(tmp_path)))
    out = tmp_path / "cli-weighted.jsonl"

    rc = mmsf.main(["--sources", "a,b", "--output", str(out), "--weighted"])

    assert rc == 0
    assert [r["id"] for r in _read_jsonl(out)] == ["a1", "b1"]
    printed = capsys.readouterr().out
    assert "[fusion] Sources: ['a', 'b']" in printed
    assert "[fusion] Output: 2 examples" in printed


def test_main_requires_sources_and_output():
    with pytest.raises(SystemExit) as exc:
        mmsf.main([])
    assert exc.value.code == 2

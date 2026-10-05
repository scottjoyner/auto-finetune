"""Tests for src.mixcorpus: blending, provenance, and the benchmark holdout."""
from __future__ import annotations

import json

import pytest

from conftest import make_cfg

from src.format_dataset import _write_dataset, read_provenance, verify_dataset
from src.mixcorpus import mix_corpus


def _src(path, n, prefix, sessions=None):
    rows = [{"messages": [{"role": "user", "content": f"{prefix}-{i}"}]}
            for i in range(n)]
    _write_dataset(str(path), rows,
                   sessions if sessions is not None else
                   [f"sess_{prefix}_{i}" for i in range(n)])


def test_mix_writes_aligned_provenance(tmp_path):
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 10, "tool")
    _src(gen, 10, "gen")
    out = tmp_path / "train.mixed.jsonl"
    n_tool, n_gen = mix_corpus(str(tool), str(out), [str(gen)],
                               general_ratio=0.5, seed=1)
    sids = read_provenance(out)
    assert sids is not None
    assert len(sids) == n_tool + n_gen == sum(1 for _ in out.open())


def test_mix_provenance_survives_shuffle_and_sampling(tmp_path):
    """The blend shuffles and downsamples, so pairing must travel with rows."""
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 20, "tool")
    _src(gen, 20, "gen")
    out = tmp_path / "train.mixed.jsonl"
    mix_corpus(str(tool), str(out), [str(gen)], general_ratio=0.25, seed=7)
    sids = read_provenance(out)
    tool_sids = {f"sess_tool_{i}" for i in range(20)}
    got = set(sids)
    # sampled general + all tool present, none invented
    assert tool_sids <= got
    assert all(s.startswith("sess_") for s in got)
    assert len(got) == len(sids)


def test_mix_holds_out_benchmark_sessions(tmp_path):
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 10, "tool", sessions=[f"sess_tool_{i}" for i in range(10)])
    _src(gen, 10, "gen")
    out = tmp_path / "train.mixed.jsonl"
    mix_corpus(str(tool), str(out), [str(gen)], general_ratio=0.5, seed=1,
               exclude={"sess_tool_3"})
    sids = read_provenance(out)
    assert "sess_tool_3" not in sids
    assert "tool-3" not in out.read_text()


def test_mix_verifies_clean_after_holdout(tmp_path):
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 10, "tool")
    _src(gen, 10, "gen")
    out = tmp_path / "train.mixed.jsonl"
    mix_corpus(str(tool), str(out), [str(gen)], general_ratio=0.5, seed=1,
               exclude={"sess_tool_0"})
    assert verify_dataset(out, {"sess_tool_0"})["status"] == "clean"


def test_mix_is_deterministic_for_a_seed(tmp_path):
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 12, "tool")
    _src(gen, 12, "gen")
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    mix_corpus(str(tool), str(a), [str(gen)], general_ratio=0.5, seed=42)
    mix_corpus(str(tool), str(b), [str(gen)], general_ratio=0.5, seed=42)
    assert a.read_text() == b.read_text()
    assert read_provenance(a) == read_provenance(b)


def test_mix_works_without_provenance_sidecars(tmp_path, capsys):
    """Legacy sources still blend; the rows just carry no session id."""
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    tool.write_text('{"messages": [{"role": "user", "content": "t"}]}\n')
    gen.write_text('{"messages": [{"role": "user", "content": "g"}]}\n')
    out = tmp_path / "train.mixed.jsonl"
    mix_corpus(str(tool), str(out), [str(gen)], general_ratio=0.5, seed=1)
    assert out.is_file()
    # a sidecar still exists, but with blank ids rather than wrong ones
    assert read_provenance(out) == ["", ""]


def test_mix_rejects_provenance_that_drifted(tmp_path):
    """A sidecar of the wrong length must not be attached to the rows."""
    tool = tmp_path / "train.combined.jsonl"
    gen = tmp_path / "general.jsonl"
    _src(tool, 5, "tool")
    _src(gen, 5, "gen")
    (tool.with_suffix(".provenance.jsonl")).write_text(
        json.dumps({"row": 0, "session_id": "only_one"}) + "\n")
    out = tmp_path / "train.mixed.jsonl"
    mix_corpus(str(tool), str(out), [str(gen)], general_ratio=0.5, seed=1)
    sids = read_provenance(out)
    assert len(sids) == 10
    # the general source's sidecar is intact; the tool side's was discarded,
    # so its 5 rows are blank rather than wrongly attributed
    assert set(sids) == {f"sess_gen_{i}" for i in range(5)} | {""}
    assert sids.count("") == 5
    assert "only_one" not in sids


def test_mix_rejects_bad_ratio(tmp_path):
    tool = tmp_path / "train.combined.jsonl"
    _src(tool, 4, "tool")
    with pytest.raises(ValueError, match="general_ratio"):
        mix_corpus(str(tool), str(tmp_path / "out.jsonl"), [],
                   general_ratio=1.5)
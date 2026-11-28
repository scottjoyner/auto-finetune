"""Focused OFFLINE tests for deterministic pure helpers in the Neo4j modules.

Scope discipline (iteration 15):
  * src/neo4j_watcher.py exposes two deterministic, filesystem-only helpers:
    ``_read_state`` and ``_write_state``. They neither contact Neo4j nor read
    credentials, so they are safe to exercise here -- including their failure
    paths (missing/corrupt files, unwritable targets, non-serializable input).
  * src/neo4j_cdc.py has NO pure helper surface: every callable either touches
    a live driver/session (connect, is_cdc_available, stream_changes,
    _stream_cdc, _poll_changes, close) or loads config/argparse (__init__,
    main). There is no query builder, parser, or filter to test without a
    database, so none is imported or exercised here (see coverage report).

No live Neo4j instance, network access, or credentials are used by this file.
"""
from __future__ import annotations

import json

import pytest

from src import neo4j_watcher as W


DEFAULT = {"trace_groups": 0, "tool_calls": 0, "last_check": 0}


# ---------------------------------------------------------------------------
# _read_state: parsing / edge cases
# ---------------------------------------------------------------------------

def test_read_state_missing_file_returns_default(tmp_path):
    assert W._read_state(str(tmp_path / "does-not-exist.json")) == DEFAULT


def test_read_state_valid_json_returned_verbatim(tmp_path):
    path = tmp_path / "state.json"
    payload = {
        "trace_groups": 100,
        "tool_calls": 50,
        "last_check": 12345.5,
        "extra_key": "kept-as-is",
    }
    path.write_text(json.dumps(payload))
    assert W._read_state(str(path)) == payload


def test_read_state_partial_json_is_not_merged_with_defaults(tmp_path):
    """The helper returns exactly what was parsed; it does not backfill keys."""
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"trace_groups": 7}))
    assert W._read_state(str(path)) == {"trace_groups": 7}


def test_read_state_non_dict_json_returned_as_is(tmp_path):
    """No type validation: a JSON list is returned unchanged, not defaulted."""
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]))
    assert W._read_state(str(path)) == [1, 2, 3]


@pytest.mark.parametrize("bad", ["not json", "", "{", "{'x': 1}"])
def test_read_state_corrupt_json_returns_default(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(bad)
    assert W._read_state(str(path)) == DEFAULT


def test_read_state_directory_path_returns_default(tmp_path):
    """A directory 'exists' but cannot be opened as a file -> default."""
    assert W._read_state(str(tmp_path)) == DEFAULT


# ---------------------------------------------------------------------------
# _write_state: serialization / edge cases
# ---------------------------------------------------------------------------

def test_write_then_read_roundtrip_preserves_types(tmp_path):
    path = str(tmp_path / "state.json")
    state = {
        "trace_groups": 9,
        "tool_calls": 13,
        "last_check": 1700000000.125,
        "nested": {"n": 1},
        "label": "caf\u00e9",
    }
    W._write_state(path, state)
    assert W._read_state(path) == state


def test_write_state_overwrites_existing_file(tmp_path):
    path = str(tmp_path / "state.json")
    W._write_state(path, {"trace_groups": 1, "tool_calls": 1, "last_check": 1.0})
    replacement = {"trace_groups": 2, "tool_calls": 2, "last_check": 2.0}
    W._write_state(path, replacement)
    assert W._read_state(path) == replacement


# ---------------------------------------------------------------------------
# _write_state: failure paths (must never raise)
# ---------------------------------------------------------------------------

def test_write_state_missing_parent_dir_swallows_error(tmp_path):
    path = str(tmp_path / "no-such-dir" / "state.json")
    W._write_state(path, DEFAULT)  # must not raise
    assert not (tmp_path / "no-such-dir" / "state.json").exists()


def test_write_state_directory_path_swallows_error(tmp_path):
    W._write_state(str(tmp_path), DEFAULT)  # must not raise (IsADirectoryError)


def test_write_state_nonserializable_input_swallows_error(tmp_path):
    """json.dump raises TypeError mid-write; the helper must stay quiet."""
    path = str(tmp_path / "state.json")
    W._write_state(path, {"trace_groups": 1, "bad": {1, 2, 3}})  # must not raise
    # A partial/truncated file is left behind; reading it falls back to default.
    assert W._read_state(path) == DEFAULT

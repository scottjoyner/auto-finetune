"""Tests for src.neo4j_watcher pure-logic helpers."""
from __future__ import annotations

import json
import os
import tempfile

from src import neo4j_watcher as W


def test_read_state_missing():
    assert W._read_state("/tmp/nonexistent_state.json") == {"trace_groups": 0, "tool_calls": 0, "last_check": 0}


def test_write_and_read_state():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "state.json")
    state = {"trace_groups": 100, "tool_calls": 50, "last_check": 12345.0}
    W._write_state(path, state)
    read = W._read_state(path)
    assert read["trace_groups"] == 100
    assert read["tool_calls"] == 50


def test_read_state_corrupt():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "state.json")
    with open(path, "w") as f:
        f.write("not json")
    assert W._read_state(path) == {"trace_groups": 0, "tool_calls": 0, "last_check": 0}

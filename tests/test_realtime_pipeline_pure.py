"""Offline tests for the deterministic pure surface of ``src.realtime_pipeline``.

The orchestrator proper (``_extract``/``_clean``/``_format``/``_should_train``/
``_train``/``_health_check``/``run_once``/``run_loop``) drives subprocesses,
clocks, sleeps and external backends, so it is deliberately out of scope here.
The only deterministic surface is the ``PipelineState`` data container and the
known-field filter plus fallback in ``RealtimePipeline._load_state``. The
latter is exercised with in-memory stubs: no real file, network, socket,
subprocess, GPU, DB or wall-clock sleep is used.
"""
from __future__ import annotations

import io
import json
from dataclasses import asdict, fields

from src import realtime_pipeline as rp
from src.realtime_pipeline import PipelineState, RealtimePipeline

# Snapshot of the public state contract. The loader's "known fields" filter is
# derived from these dataclass fields, so a drift here is worth failing on.
EXPECTED_DEFAULTS = {
    "last_extraction": 0.0,
    "last_clean": 0.0,
    "last_format": 0.0,
    "last_train": 0.0,
    "last_health_check": 0.0,
    "total_extracted": 0,
    "total_cleaned": 0,
    "total_formatted": 0,
    "total_trained": 0,
    "new_since_train": 0,
    "runs_completed": 0,
    "runs_failed": 0,
    "current_phase": "idle",
    "last_error": None,
    "min_new_for_train": 50,
    "min_interval_train": 3600,
}


def _bare_pipeline():
    """A RealtimePipeline with no __init__ side effects (no cfg, no state I/O)."""
    obj = object.__new__(RealtimePipeline)
    obj.state_path = "state.json"
    return obj


def _state_from_payload(monkeypatch, payload: str) -> PipelineState:
    """Drive _load_state over an in-memory document; never touches the disk."""
    monkeypatch.setattr(rp.os.path, "exists", lambda _path: True)
    monkeypatch.setattr("builtins.open", lambda *_a, **_k: io.StringIO(payload))
    return _bare_pipeline()._load_state()


# --- PipelineState: pure state container -----------------------------------

def test_pipeline_state_defaults_match_field_set():
    state = PipelineState()
    data = asdict(state)
    assert data == EXPECTED_DEFAULTS
    assert set(data) == {f.name for f in fields(PipelineState)}
    # Threshold defaults are part of the contract, not incidental.
    assert state.min_new_for_train == 50
    assert state.min_interval_train == 3600
    assert state.last_error is None
    assert state.current_phase == "idle"


def test_pipeline_state_asdict_roundtrip_preserves_every_field():
    state = PipelineState(
        last_extraction=1.5,
        total_extracted=42,
        new_since_train=73,
        current_phase="training",
        last_error="boom",
        min_new_for_train=7,
        min_interval_train=12.5,
    )
    data = asdict(state)
    assert data == {
        **EXPECTED_DEFAULTS,
        "last_extraction": 1.5,
        "total_extracted": 42,
        "new_since_train": 73,
        "current_phase": "training",
        "last_error": "boom",
        "min_new_for_train": 7,
        "min_interval_train": 12.5,
    }
    # asdict() output is valid constructor input: the state is serializable.
    assert asdict(PipelineState(**data)) == data
    assert state is not PipelineState(**data)


# --- _load_state: known-field filter and fallback --------------------------

def test_load_state_missing_file_returns_defaults(monkeypatch):
    def _must_not_open(*_a, **_k):
        raise AssertionError("missing state file must not be opened")

    monkeypatch.setattr(rp.os.path, "exists", lambda _path: False)
    monkeypatch.setattr("builtins.open", _must_not_open)
    assert asdict(_bare_pipeline()._load_state()) == EXPECTED_DEFAULTS


def test_load_state_filters_unknown_fields_and_fills_defaults(monkeypatch):
    payload = json.dumps({
        "new_since_train": 12,
        "current_phase": "formatting",
        "last_error": "partial",
        "min_new_for_train": 9,
        "not_a_field": "drop me",
        "extra": [1, 2, 3],
    })
    state = _state_from_payload(monkeypatch, payload)
    assert asdict(state) == {
        **EXPECTED_DEFAULTS,
        "new_since_train": 12,
        "current_phase": "formatting",
        "last_error": "partial",
        "min_new_for_train": 9,
    }
    assert not hasattr(state, "not_a_field")
    assert not hasattr(state, "extra")


def test_load_state_corrupt_json_falls_back_to_defaults(monkeypatch):
    # json.load raises inside the try; the except path must return a clean state.
    state = _state_from_payload(monkeypatch, "{not json")
    assert asdict(state) == EXPECTED_DEFAULTS


def test_load_state_empty_document_falls_back_to_all_defaults(monkeypatch):
    assert asdict(_state_from_payload(monkeypatch, "{}")) == EXPECTED_DEFAULTS


def test_load_state_preserves_value_types_without_coercion(monkeypatch):
    payload = json.dumps({
        "last_extraction": 1.25,
        "total_trained": 5,
        "current_phase": "idle",
        "last_error": "",
        "min_interval_train": 0,
    })
    state = _state_from_payload(monkeypatch, payload)
    assert state.last_extraction == 1.25
    assert isinstance(state.last_extraction, float)
    assert state.total_trained == 5
    assert isinstance(state.total_trained, int)
    assert state.last_error == ""
    assert state.min_interval_train == 0
    assert asdict(state) == {
        **EXPECTED_DEFAULTS,
        "last_extraction": 1.25,
        "total_trained": 5,
        "last_error": "",
        "min_interval_train": 0,
    }


def test_load_state_reads_only_the_configured_state_path(monkeypatch):
    calls = []

    def _fake_open(path, *_a, **_k):
        calls.append(path)
        return io.StringIO("{}")

    monkeypatch.setattr(rp.os.path, "exists", lambda p: p == "state.json")
    monkeypatch.setattr("builtins.open", _fake_open)
    pipeline = _bare_pipeline()
    pipeline.state_path = "state.json"
    pipeline._load_state()
    assert calls == ["state.json"]

"""Regression guard: the retired recovery latch must stay retired.

Operator-approved policy (2026-11-25) removed the independent recovery latch
from xwing training admission. These tests fail if that gate is silently
reintroduced, while preserving the stop-file and memory/resident gates.
"""
import json

import pytest

from src import training_admission as a

GIB = 1024 * 1024
IDLE_INVENTORY = [{"type": "embedding", "loaded_instances": []}]
RESIDENT_LLM = IDLE_INVENTORY + [
    {"type": "llm", "loaded_instances": [{"id": "ornith"}]},
]


class _FakeResponse:
    status = 200

    def __init__(self, models):
        self._models = models

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size):
        return json.dumps({"models": self._models}).encode()


class _FakeOpener:
    def __init__(self, models):
        self._models = models

    def open(self, _url, timeout=3):
        return _FakeResponse(self._models)


@pytest.fixture
def admission(monkeypatch, tmp_path):
    """Synthetic xwing telemetry with no latch, a healthy host, idle inference."""
    monkeypatch.setattr(a.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(a, "STOP", tmp_path / "launch-next.stop")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemTotal: 32 GiB\nMemAvailable: {32 * GIB} kB\n")
    monkeypatch.setattr(a, "MEMINFO", meminfo)
    pressure = tmp_path / "pressure"
    pressure.write_text("some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
                        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    monkeypatch.setattr(a, "PRESSURE", pressure)
    monkeypatch.setattr(a, "build_opener",
                        lambda *args, **kwargs: _FakeOpener(IDLE_INVENTORY))
    return a


def test_retired_recovery_latch_is_not_in_the_module():
    """Removing the latch API is the guard against silent reintroduction."""
    assert not hasattr(a, "RECOVERY_LATCH")
    assert not hasattr(a, "_independent_recovery_latch")


def test_missing_latch_does_not_block_admission(admission):
    assert admission.read_live() == a.Decision(True, "admitted")


def test_stop_file_still_blocks_admission(admission, tmp_path):
    a.STOP.write_text("STOP")
    assert admission.read_live() == a.Decision(False, "stop_file")


def test_low_memory_still_blocks_admission(admission):
    a.MEMINFO.write_text(f"MemTotal: 32 GiB\nMemAvailable: {GIB} kB\n")
    assert admission.read_live() == a.Decision(False, "insufficient_memory")


def test_resident_llm_still_blocks_admission(admission, monkeypatch):
    monkeypatch.setattr(a, "build_opener",
                        lambda *args, **kwargs: _FakeOpener(RESIDENT_LLM))
    assert admission.read_live() == a.Decision(False, "resident_inference")

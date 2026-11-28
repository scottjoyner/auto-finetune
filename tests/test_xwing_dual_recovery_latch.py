"""Regression guard: the retired recovery latch must stay retired.

Operator-approved policy (2026-11-25) removed the independent recovery latch
from xwing training admission. These tests fail if that gate is silently
reintroduced, while preserving the stop-file and memory/resident gates.
"""
import json

import pytest

from src import training_admission as a


class _FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, _size):
        return json.dumps({"models": []}).encode()


class _FakeOpener:
    def open(self, _url, timeout=3):
        return _FakeResponse()


@pytest.fixture
def admission(monkeypatch, tmp_path):
    monkeypatch.setattr(a.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(a, "STOP", tmp_path / "launch-next.stop")
    monkeypatch.setattr(a, "RECOVERY_LATCH", tmp_path / "private" / "hold")
    monkeypatch.setattr(a, "build_opener", lambda *args, **kwargs: _FakeOpener())
    monkeypatch.setattr(a, "evaluate", lambda *args, **kwargs: a.Decision(True, "admitted"))

    def _latch_must_not_be_consulted():
        raise AssertionError("retired recovery latch was consulted")

    monkeypatch.setattr(a, "_independent_recovery_latch", _latch_must_not_be_consulted)
    return a


def test_missing_latch_does_not_block_admission(admission):
    assert admission.read_live() == a.Decision(True, "admitted")


def test_stop_file_still_blocks_admission(admission, tmp_path):
    a.STOP.write_text("STOP")
    assert admission.read_live() == a.Decision(False, "stop_file")

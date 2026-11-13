"""Real-time quality gate fail-closed acceptance, all dependencies stubbed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import realtime_pipeline as realtime


class Cfg:
    def path(self, key):
        if key == "dataset_dir":
            return "/synthetic/isolated"
        return "/synthetic/isolated"


def pipeline():
    p=object.__new__(realtime.RealtimePipeline)
    p.cfg=Cfg()
    p.state=realtime.PipelineState()
    p._save_state=lambda: None
    p._run_cmd=lambda *a, **kw: (0, "")
    p._count_cleaned_files=lambda: 3
    return p


@pytest.mark.parametrize("failure", [
    ValueError("quality_input_malformed"),
    OSError("quality output unavailable"),
    RuntimeError("synthetic scoring unavailable"),
])
def test_quality_gate_unavailable_blocks_pipeline(monkeypatch, failure):
    p=pipeline()
    monkeypatch.setattr(
        realtime, "quality_gate",
        lambda *a, **kw: (_ for _ in ()).throw(failure),
    )
    monkeypatch.setattr(
        realtime, "AlertManager",
        lambda *a, **kw: SimpleNamespace(fire=lambda *a, **kw: None),
    )
    assert p._clean() is False
    assert p.state.last_error=="quality_gate_unavailable"
    assert p.state.total_cleaned==3


def test_more_than_half_bad_quality_is_hold(monkeypatch):
    p=pipeline()
    alerts=[]
    monkeypatch.setattr(
        realtime,"quality_gate",
        lambda *a, **kw: {
            "passed":1, "failed":3, "total":4, "avg_score":.3,
        },
    )
    monkeypatch.setattr(
        realtime,"AlertManager",
        lambda *a, **kw: SimpleNamespace(
            fire=lambda *args: alerts.append(args),
        ),
    )
    assert p._clean() is False
    assert p.state.last_error=="quality_gate_failure_rate"
    assert alerts


def test_successful_quality_gate_allows_cleaning(monkeypatch):
    p=pipeline()
    monkeypatch.setattr(
        realtime,"quality_gate",
        lambda *a, **kw: {
            "passed":4, "failed":0, "total":4, "avg_score":.99,
        },
    )
    monkeypatch.setattr(
        realtime, "AlertManager",
        lambda *a, **kw: SimpleNamespace(
            fire=lambda *a, **kw: pytest.fail("unexpected quality alert"),
        ),
    )
    assert p._clean() is True
    assert p.state.total_cleaned==3


def test_failed_clean_subprocess_blocks_before_quality_gate(monkeypatch):
    p=pipeline()
    p._run_cmd=lambda *a, **kw: (78, "synthetic refusal")
    monkeypatch.setattr(
        realtime, "quality_gate",
        lambda *a, **kw: pytest.fail("quality gate after failed cleaner"),
    )
    assert p._clean() is False
    assert p.state.last_error=="synthetic refusal"


def test_failed_quality_prevents_next_pipeline_stage(monkeypatch):
    p=pipeline()
    p._extract=lambda: True
    p._clean=lambda: False
    p._format=lambda: pytest.fail("format executed after quality HOLD")
    p._train=lambda: pytest.fail("train executed after quality HOLD")
    p._health_check=lambda: {"status":"synthetic_unknown"}
    result=p.run_once()
    assert result["extraction"] is True
    assert result["cleaning"] is False
    assert result["formatting"] is False
    assert result["training"] is False

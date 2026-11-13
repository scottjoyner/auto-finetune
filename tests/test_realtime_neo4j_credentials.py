"""Credential-free, synthetic tests for realtime Neo4j extraction safeguards."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import realtime_pipeline as realtime


def fixture_pipeline():
    obj = object.__new__(realtime.RealtimePipeline)
    obj.state = realtime.PipelineState()
    obj.repo = "/synthetic/checkout"
    saves = []
    obj._save_state = lambda: saves.append(obj.state.last_error)
    obj._count_raw_files = lambda: 4
    return obj, saves


@pytest.mark.parametrize("value", [None, ""])
def test_missing_credential_blocks_before_child_or_data_mutation(monkeypatch, value):
    pipe, saves = fixture_pipeline()
    if value is None:
        monkeypatch.delenv("NEO4J_TRACES_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("NEO4J_TRACES_PASSWORD", value)

    def forbidden(*args, **kwargs):
        pytest.fail("child process or dataset scan reached with no credential")

    pipe._run_cmd = forbidden
    pipe._count_raw_files = forbidden
    assert pipe._extract() is False
    assert pipe.state.last_error == "neo4j_traces_credential_unavailable"
    assert pipe.state.total_extracted == 0
    assert pipe.state.new_since_train == 0
    assert saves[-1] == "neo4j_traces_credential_unavailable"


def test_present_runtime_credential_stays_out_of_overrides(monkeypatch):
    pipe, _ = fixture_pipeline()
    secret = "synthetic-pass-only"
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", secret)
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return 0, ""

    pipe._run_cmd = fake_run
    assert pipe._extract() is True
    assert len(observed) == 1
    command, kwargs = observed[0]
    assert command[-1] == "--incremental"
    assert kwargs["extra_env"] == {"ROCM_PATH": "/opt/rocm"}
    assert secret not in repr(command)
    assert secret not in repr(kwargs)
    assert pipe.state.total_extracted == 4
    assert pipe.state.new_since_train == 4


def test_failed_extractor_does_not_mark_extraction_success(monkeypatch):
    pipe, _ = fixture_pipeline()
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", "synthetic-pass-only")
    pipe._run_cmd = lambda *a, **k: (1, "synthetic-access-denied")
    pipe._count_raw_files = lambda: pytest.fail("raw files counted after failure")
    monkeypatch.setattr(
        realtime, "AlertManager",
        lambda *a, **k: type("A", (), {"fire": lambda *args: None})(),
    )
    pipe.cfg = object()
    assert pipe._extract() is False
    assert pipe.state.last_error == "synthetic-access-denied"
    assert pipe.state.total_extracted == 0


def test_subprocess_inherits_only_runtime_secret_and_scrubs_output(monkeypatch):
    pipe, _ = fixture_pipeline()
    secret = "FAKE_TEST_SECRET_928371"
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", secret)
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED", "1")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, "error " + secret, "\n" + secret)

    monkeypatch.setattr(realtime.subprocess, "run", fake_run)
    code, output = pipe._run_cmd(["python", "-m", "src.cli", "neo4j-traces"])
    assert code == 1
    assert "[REDACTED]" in output
    assert secret not in output
    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert secret not in repr(cmd)
    assert kwargs["env"]["NEO4J_TRACES_PASSWORD"] == secret
    assert "AF_SKIP_UNMANAGED_CHECK" not in kwargs["env"]
    assert "TRAIN_ALLOW_UNVERIFIED" not in kwargs["env"]


def test_redaction_precedes_tail_truncation(monkeypatch):
    pipe, _ = fixture_pipeline()
    secret = "long_synthetic_secret"
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", secret)
    monkeypatch.setattr(
        realtime.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 1, "x" * 2100 + secret + "y" * 100, "",
        ),
    )
    _, output = pipe._run_cmd(["read-only-fake"])
    assert secret not in output
    assert "[REDACTED]" in output
    assert len(output) <= 2000


def test_source_does_not_embed_password_literal():
    source = (Path(__file__).resolve().parents[1] /
              "src/realtime_pipeline.py").read_text()
    assert '"NEO4J_TRACES_PASSWORD": "' not in source
    assert '"NEO4J_TRACES_PASSWORD":" ' not in source
    assert 'neo4j_traces_credential_unavailable' in source


@pytest.mark.parametrize(
    ("result", "exit_code"),
    [
        ({"extraction": False, "cleaning": False, "formatting": False}, 78),
        ({"extraction": True, "cleaning": False, "formatting": False}, 1),
        ({"extraction": True, "cleaning": True, "formatting": False}, 1),
        ({"extraction": True, "cleaning": True, "formatting": True,
          "training": False}, 0),
    ],
)
def test_realtime_main_exit_reflects_pipeline_failure(monkeypatch, result, exit_code):
    class FakePipeline:
        def __init__(self, cfg):
            self.state = realtime.PipelineState()
        def run_once(self):
            return result
    monkeypatch.setattr(realtime, "load", lambda: object())
    monkeypatch.setattr(realtime, "RealtimePipeline", FakePipeline)
    assert realtime.main(["--mode=once"]) == exit_code


def test_run_once_missing_credential_does_not_reach_later_stages(monkeypatch):
    pipe, _ = fixture_pipeline()
    monkeypatch.delenv("NEO4J_TRACES_PASSWORD", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("workload stage executed following missing credential HOLD")
    pipe._run_cmd = forbidden
    pipe._clean = forbidden
    pipe._format = forbidden
    pipe._train = forbidden
    pipe._health_check = lambda: {"status": "synthetic_unknown"}
    result = pipe.run_once()
    assert result["extraction"] is False
    assert result["cleaning"] is False
    assert result["formatting"] is False
    assert result["training"] is False
    assert result["health"]["status"] == "synthetic_unknown"
    assert pipe.state.last_error == "neo4j_traces_credential_unavailable"
    assert pipe.state.runs_completed == 0


def test_subprocess_exception_does_not_echo_runtime_secret(monkeypatch):
    pipe, _ = fixture_pipeline()
    secret = "EXCEPTION_ONLY_SYNTHETIC_SECRET"
    monkeypatch.setenv("NEO4J_TRACES_PASSWORD", secret)
    def broken(*args, **kwargs):
        raise OSError("connection failed " + secret)
    monkeypatch.setattr(realtime.subprocess, "run", broken)
    code, detail = pipe._run_cmd(["synthetic"])
    assert code == 1
    assert secret not in detail
    assert "[REDACTED]" in detail

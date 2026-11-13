"""Synthetic fail-closed regression tests for the former training-bypass paths.

No model loads, dataset writes, GPU inspection, service calls or credential access.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src import cli, train, training_admission as admission
from src.config import Config, _DEFAULTS


@pytest.mark.parametrize("legacy", ["1", "0", "anything"])
def test_xwing_missing_recovery_latch_cannot_be_bypassed(
    monkeypatch, tmp_path, legacy
):
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", legacy)
    monkeypatch.setattr(admission.socket, "gethostname", lambda: "xwing")
    monkeypatch.setattr(admission, "STOP", tmp_path / "absent-stop")
    monkeypatch.setattr(admission, "RECOVERY_LATCH", tmp_path / "absent-latch")
    result = admission.read_live()
    assert not result.allowed
    assert result.reason == "recovery_latch_missing"


def test_direct_train_does_not_bypass_when_env_requests_skip(monkeypatch):
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(
        admission, "read_live",
        lambda: admission.Decision(False, "recovery_latch_missing"),
    )
    monkeypatch.setattr(
        train, "_reject_held_out",
        lambda *a, **k: pytest.fail("dataset accessed before admission"),
    )
    assert train.main(cfg=None) == 78


@pytest.mark.parametrize("cmd", ["train", "dpo"])
def test_cli_does_not_bypass_when_env_requests_skip(monkeypatch, cmd):
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(
        admission, "read_live",
        lambda: admission.Decision(False, "recovery_latch"),
    )
    monkeypatch.setattr(
        cli, "load", lambda: pytest.fail("configuration loaded before admission")
    )
    assert cli.main(["cli", cmd]) == 78


def test_cli_admission_exception_fails_closed(monkeypatch):
    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(
        admission, "read_live",
        lambda: (_ for _ in ()).throw(RuntimeError("telemetry unavailable")),
    )
    monkeypatch.setattr(
        cli, "load", lambda: pytest.fail("configuration loaded before admission")
    )
    assert cli.main(["cli", "train"]) == 78


@pytest.mark.parametrize("error", [False, True])
def test_unmanaged_trainer_census_cannot_be_bypassed(monkeypatch, error):
    from src import locking

    monkeypatch.setenv("AF_SKIP_UNMANAGED_CHECK", "1")
    monkeypatch.setattr(
        admission, "read_live", lambda: admission.Decision(True, "synthetic_admitted")
    )
    monkeypatch.setattr(
        cli, "load", lambda: Config(raw=copy.deepcopy(_DEFAULTS))
    )

    def fake_census(*args, **kwargs):
        if error:
            raise OSError("census unavailable")
        return [{"pid": 12345}]

    monkeypatch.setattr(locking, "unmanaged_training_processes", fake_census)
    monkeypatch.setattr(
        locking, "command_leases",
        lambda *a, **k: pytest.fail("lease reached despite unknown GPU owner"),
    )
    assert cli.main(["cli", "train"]) == 75


def test_no_env_bypass_in_published_entrypoints():
    root = Path(__file__).resolve().parents[1]
    for name in ("src/training_admission.py", "src/train.py", "src/cli.py"):
        source = (root / name).read_text()
        assert "AF_SKIP_UNMANAGED_CHECK" not in source
        assert "admission_bypassed" not in source
    assert 'setdefault("AF_SKIP_UNMANAGED_CHECK"' not in (
        root / "tests/conftest.py"
    ).read_text()


def test_child_launchers_strip_inherited_bypass():
    root = Path(__file__).resolve().parents[1]
    for name in ("scripts/train_buckets.py", "src/scheduled_retrain.py",
                 "src/realtime_pipeline.py"):
        source = (root / name).read_text()
        for key in ("TRAIN_ALLOW_UNVERIFIED", "AF_SKIP_UNMANAGED_CHECK"):
            assert f'env.pop("{key}", None)' in source
            assert f'env["{key}"] = "1"' not in source
            assert f'"{key}": "1"' not in source
    assert 'cwd=str(Path(__file__).resolve().parents[1])' in (
        root / "scripts/train_buckets.py"
    ).read_text()

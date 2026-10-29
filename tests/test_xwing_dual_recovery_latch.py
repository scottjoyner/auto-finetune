"""Synthetic source/missing/untrusted hold files; no network or trainer run."""
from pathlib import Path
import os

import pytest

from src import training_admission as a


@pytest.fixture
def paths(tmp_path,monkeypatch):
    # Never inherit a repository-wide test bypass when exercising recovery HOLD.
    monkeypatch.delenv("AF_SKIP_UNMANAGED_CHECK", raising=False)
    parent=tmp_path/"private-custody"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    stop=tmp_path/"launch-next.stop"
    latch=parent/"qwen-training-recovery.hold"
    monkeypatch.setattr(a.socket,"gethostname",lambda:"xwing")
    monkeypatch.setattr(a,"STOP",stop)
    monkeypatch.setattr(a,"RECOVERY_LATCH",latch)
    return stop,latch


def test_stop_priority_unchanged(paths):
    stop,latch=paths
    stop.write_text("STOP")
    assert a.read_live()==a.Decision(False,"stop_file")


def test_missing_primary_stop_cannot_be_training_admission(paths):
    stop,latch=paths
    latch.write_text("DENY")
    latch.chmod(0o600)
    assert a.read_live()==a.Decision(False,"recovery_latch")


def test_both_hold_files_missing_denies_even_with_unobserved_resources(paths):
    assert a.read_live()==a.Decision(False,"recovery_latch_missing")


def test_missing_parent_denies(paths):
    stop,latch=paths
    latch.parent.rmdir()
    assert a.read_live()==a.Decision(False,"recovery_latch_missing")


def test_world_readable_latch_denied(paths):
    stop,latch=paths
    latch.write_text("DENY")
    latch.chmod(0o644)
    assert a.read_live()==a.Decision(False,"recovery_latch_untrusted")


def test_symlink_latch_denied(paths):
    stop,latch=paths
    elsewhere=latch.parent/"other"
    elsewhere.write_text("safe looking")
    latch.symlink_to(elsewhere)
    assert a.read_live()==a.Decision(False,"recovery_latch_untrusted")


def test_symlink_primary_stop_itself_denies(paths):
    stop,latch=paths
    stop.symlink_to("/file/that/does/not/exist")
    assert a.read_live()==a.Decision(False,"stop_file")


def test_untrusted_private_parent_denied(paths):
    stop,latch=paths
    latch.write_text("DENY")
    latch.chmod(0o600)
    latch.parent.chmod(0o755)
    assert a.read_live()==a.Decision(False,"recovery_latch_parent_untrusted")


def test_non_xwing_host_still_unchanged(paths,monkeypatch):
    monkeypatch.setattr(a.socket,"gethostname",lambda:"x1-370")
    assert a.read_live()==a.Decision(True,"other_host")


def test_latch_has_no_implicit_release_on_empty_contents(paths):
    stop,latch=paths
    latch.touch(mode=0o600)
    latch.chmod(0o600)
    assert a.read_live()==a.Decision(False,"recovery_latch")


def test_direct_guard_cli_exits_78_when_stop_missing(paths,capsys):
    stop,latch=paths
    latch.write_text("DENY")
    latch.chmod(0o600)
    assert a.main()==78
    assert "[train-admission] HOLD: recovery_latch" in capsys.readouterr().out


def test_no_stop_or_latch_mutation_by_guard(paths):
    stop,latch=paths
    latch.write_text("DENY")
    latch.chmod(0o600)
    before=latch.stat().st_mtime_ns
    a.read_live()
    assert latch.stat().st_mtime_ns==before
    assert not stop.exists()

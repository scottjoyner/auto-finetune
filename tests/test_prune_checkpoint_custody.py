"""Prune-safety tests on disposable files ONLY; never touch real checkpoints."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import time

from scripts import prune_checkpoints as prune


def _layout(tmp_path: Path, monkeypatch, stop: bool = False):
    base = tmp_path / "checkpoints"
    base.mkdir()
    label = base / "toolcall-model"
    label.mkdir()
    for index in (10, 20, 30, 40):
        ckpt = label / ("checkpoint-" + str(index))
        ckpt.mkdir()
        for f in prune.RESUME_FILES:
            (ckpt / f).write_text("irrecoverable-resume-state")
        (ckpt / "model.safetensors").write_text("weights")
    (label / "adapter_model.safetensors").write_text("FINAL ADAPTER")
    os.utime(label, (time.time() - 12*3600, time.time() - 12*3600))
    stop_path = tmp_path / "launch-next.stop"
    if stop:
        stop_path.write_text("recovery hold")
    monkeypatch.setattr(prune, "STOP_FILE", str(stop_path))
    monkeypatch.setattr(prune, "BASE", str(base))
    return base, label


def _call(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prune_checkpoints.py", *args])
    return prune.main()


def test_recovery_stopfile_denies_all_apply_before_enumeration(tmp_path, monkeypatch, capsys):
    _layout(tmp_path, monkeypatch, stop=True)
    for flags in (("--apply",), ("--apply", "--aggressive"), ("--apply", "--keep", "other")):
        assert _call(monkeypatch, *flags) == 78
        assert "HOLD" in capsys.readouterr().out
    assert len(list((tmp_path / "checkpoints" / "toolcall-model").rglob("optimizer.pt"))) == 4


def test_standard_apply_only_old_checkpoint_resume_files(tmp_path, monkeypatch):
    _, label = _layout(tmp_path, monkeypatch)
    assert _call(monkeypatch, "--apply") == 0
    for idx in (10, 20):
        old = label / f"checkpoint-{idx}"
        assert old.is_dir()
        assert not any((old / item).exists() for item in prune.RESUME_FILES)
        assert (old / "model.safetensors").is_file()
    for idx in (30, 40):
        recent = label / f"checkpoint-{idx}"
        assert all((recent / item).is_file() for item in prune.RESUME_FILES)
        assert (recent / "model.safetensors").is_file()
    assert (label / "adapter_model.safetensors").read_text() == "FINAL ADAPTER"


def test_aggressive_apply_only_old_checkpoint_dirs(tmp_path, monkeypatch):
    _, label = _layout(tmp_path, monkeypatch)
    assert _call(monkeypatch, "--apply", "--aggressive") == 0
    assert not (label / "checkpoint-10").exists()
    assert not (label / "checkpoint-20").exists()
    for idx in (30, 40):
        assert all((label / f"checkpoint-{idx}" / f).exists() for f in prune.RESUME_FILES)
    assert (label / "adapter_model.safetensors").is_file()


def test_dry_run_cannot_delete_any_checkpoint_files(tmp_path, monkeypatch):
    _, label = _layout(tmp_path, monkeypatch)
    assert _call(monkeypatch) == 0
    assert len(list(label.rglob("optimizer.pt"))) == 4
    assert _call(monkeypatch, "--aggressive") == 0
    assert len(list(label.rglob("optimizer.pt"))) == 4


def test_keep_preserves_entire_label_even_on_apply(tmp_path, monkeypatch):
    _, label = _layout(tmp_path, monkeypatch)
    assert _call(monkeypatch, "--apply", "--keep", "toolcall-model") == 0
    assert len(list(label.rglob("optimizer.pt"))) == 4


def test_unrecognized_checkpoint_names_and_symlinks_not_pruned(tmp_path, monkeypatch):
    _, label = _layout(tmp_path, monkeypatch)
    unknown = label / "checkpoint-rollback-rescue"
    unknown.mkdir()
    (unknown / "optimizer.pt").write_text("RESCUE")
    linked = label / "checkpoint-5"
    linked.symlink_to(label / "checkpoint-30", target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "optimizer.pt").write_text("OUTSIDE")
    (label / "checkpoint-7").symlink_to(outside, target_is_directory=True)
    assert _call(monkeypatch, "--apply", "--aggressive") == 0
    assert (unknown / "optimizer.pt").read_text() == "RESCUE"
    assert linked.is_symlink()
    assert (outside / "optimizer.pt").read_text() == "OUTSIDE"


def test_never_follow_symlinked_label_directory(tmp_path, monkeypatch):
    base, label = _layout(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "checkpoint-1").mkdir()
    (outside / "checkpoint-1" / "optimizer.pt").write_text("OUTSIDE")
    (base / "external-link").symlink_to(outside, target_is_directory=True)
    assert _call(monkeypatch, "--apply", "--aggressive") == 0
    assert (outside / "checkpoint-1" / "optimizer.pt").read_text() == "OUTSIDE"

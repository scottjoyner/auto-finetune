"""Tests for opt-in SSD Cargo target placement. Never build a real model."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research_storage_preflight import GIB
from src import ssd_cargo_runner as s


@pytest.fixture
def fake_mounted_ssd(tmp_path, monkeypatch):
    repo = tmp_path / "my-rust-repo"
    repo.mkdir()
    (repo / "Cargo.toml").write_text('[package]\nname="tiny"\nversion="0.1.0"\n')
    mount = tmp_path / "ssd-mount"
    mount.mkdir()
    cargo = tmp_path / "cargo"
    cargo.write_text("#!/bin/sh\nexit 0\n")
    cargo.chmod(0o700)
    original_is_mount = Path.is_mount
    monkeypatch.setattr(Path, "is_mount",
                        lambda self: self == mount or original_is_mount(self))
    monkeypatch.setattr(s, "_device",
                        lambda path: 277 if Path(path) == mount or
                        Path(path).is_relative_to(mount) else 101)
    monkeypatch.setattr(s, "_available_bytes",
                        lambda path: (200 if Path(path) == mount else 25) * GIB)
    return repo, mount, cargo


def test_plan_is_read_only_and_targets_unique_ssd_worktree(fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo)
    assert job["can_launch"]
    assert job["root_device"] != job["ssd_device"]
    assert job["cargo_target_dir"].startswith(str(mount / s.NAMESPACE))
    assert not (mount / s.NAMESPACE).exists()
    assert job["cargo_incremental"] == "0"
    assert job["space_reserved"] is False
    assert job["directories_created"] is False
    assert "--offline" in job["cargo_command"]
    assert "--locked" in job["cargo_command"]
    assert job == s.plan(repo, mount, cargo_binary=cargo)
    assert not (repo / "target").exists()


def test_distinct_repos_never_share_artifact_target(fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    other = repo.parent / "another-rust-repo"
    other.mkdir()
    (other / "Cargo.toml").write_text('[workspace]\n')
    first = s.plan(repo, mount, cargo_binary=cargo)
    second = s.plan(other, mount, cargo_binary=cargo)
    assert first["cargo_target_dir"] != second["cargo_target_dir"]


@pytest.mark.parametrize("limits", [
    {"root_floor_gib": 30},
    {"ssd_floor_gib": 300},
    {"planned_write_gib": 180, "ssd_reserve_gib": 30},
])
def test_capacity_fail_closed_without_creating_anything(fake_mounted_ssd, limits):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo, **limits)
    assert job["can_launch"] is False
    with pytest.raises(s.CargoCapacityBlocked):
        s.run_explicit(job)
    assert not (mount / s.NAMESPACE).exists()


def test_explicit_run_targets_ssd_and_does_not_mutate_repo(
        fake_mounted_ssd, monkeypatch):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo)
    calls = []
    def fake_execute(args, *, cwd, env, check):
        calls.append((args, cwd, env, check))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(s.subprocess, "run", fake_execute)
    assert s.run_explicit(job) == 0
    assert len(calls) == 1
    argv, cwd, env, check = calls[0]
    assert cwd == str(repo)
    assert argv[0] == str(cargo)
    assert env["CARGO_TARGET_DIR"] == job["cargo_target_dir"]
    assert env["CARGO_INCREMENTAL"] == "0"
    assert env["RUSTC"] == str(s.DEFAULT_RUSTC)
    assert env["TMPDIR"] == job["ssd_tmpdir"]
    assert Path(env["CARGO_TARGET_DIR"]).is_dir()
    assert Path(env["TMPDIR"]).is_dir()
    assert not (repo / "target").exists()
    assert check is False


def test_recheck_blocks_after_capacity_changes(fake_mounted_ssd, monkeypatch):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo)
    monkeypatch.setattr(s, "_available_bytes",
                        lambda path: (200 if Path(path) == mount else 19) * GIB)
    with pytest.raises(s.CargoCapacityBlocked, match="changed"):
        s.run_explicit(job)
    assert not (mount / s.NAMESPACE).exists()


def test_symlinked_manifest_or_target_rejected(fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    alias = repo.parent / "alias-repo"
    alias.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        s.plan(alias, mount, cargo_binary=cargo)
    target_parent = mount / s.NAMESPACE
    target_parent.symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        s.plan(repo, mount, cargo_binary=cargo)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1, True, "30"])
def test_invalid_limits_rejected(fake_mounted_ssd, invalid):
    repo, mount, cargo = fake_mounted_ssd
    with pytest.raises(ValueError):
        s.plan(repo, mount, cargo_binary=cargo, ssd_reserve_gib=invalid)


def test_manifest_and_command_validation(fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    with pytest.raises(ValueError, match="unsupported"):
        s.plan(repo, mount, cargo_binary=cargo, command="clean")
    (repo / "Cargo.toml").unlink()
    with pytest.raises(ValueError, match="Cargo.toml"):
        s.plan(repo, mount, cargo_binary=cargo)


def test_missing_compiler_fails_before_target_creation(fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo,
                 rustc_binary=mount / "absent-rustc")
    with pytest.raises(ValueError, match="Rust compiler"):
        s.run_explicit(job)
    assert not (mount / s.NAMESPACE).exists()


def test_mount_device_changed_after_plan_is_blocked(fake_mounted_ssd, monkeypatch):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo)
    monkeypatch.setattr(s, "_device", lambda _: job["root_device"])
    with pytest.raises(s.CargoCapacityBlocked, match="mount identity"):
        s.run_explicit(job)
    assert not (mount / s.NAMESPACE).exists()


def test_unmounted_or_root_device_never_accepted(fake_mounted_ssd, monkeypatch):
    repo, mount, cargo = fake_mounted_ssd
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    with pytest.raises(ValueError, match="actual mounted"):
        s.plan(repo, mount, cargo_binary=cargo)
    monkeypatch.setattr(Path, "is_mount", lambda _: True)
    monkeypatch.setattr(s, "_device", lambda _: 101)
    with pytest.raises(ValueError, match="same filesystem as root"):
        s.plan(repo, mount, cargo_binary=cargo)


def test_release_command_explicit_and_manifest_symlink_rejected(
        fake_mounted_ssd):
    repo, mount, cargo = fake_mounted_ssd
    job = s.plan(repo, mount, cargo_binary=cargo, release=True,
                 command="build")
    assert job["cargo_command"][1] == "build"
    assert job["cargo_command"][-1] == "--release"
    assert not (mount / s.NAMESPACE).exists()
    manifest = repo / "Cargo.toml"
    source = repo / "real.toml"
    manifest.rename(source)
    manifest.symlink_to(source)
    with pytest.raises(ValueError, match="manifest cannot be a symlink"):
        s.plan(repo, mount, cargo_binary=cargo)

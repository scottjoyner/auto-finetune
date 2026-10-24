"""Synthetic pyvenv layouts; no training package imports or pip operations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts import qwen_python_runtime_audit as a


def fixture(tmp_path, *, python_version="3.12.3", current="3.14", has_original=True):
    venv=tmp_path/"venv"
    (venv/"bin").mkdir(parents=True)
    executable=tmp_path/"original-python"
    if has_original:
        executable.write_text("fixture binary marker")
    (venv/"pyvenv.cfg").write_text(
        f"home = /usr/bin\nversion = {python_version}\nexecutable = {executable}\n"
    )
    old_site=venv/"lib"/"python3.12"/"site-packages"
    old_site.mkdir(parents=True)
    for name in a.NEEDED:
        (old_site/name).mkdir()
        (old_site/f"{name}-1.0.dist-info").mkdir()
    return venv, {"major":3,"minor":int(current.split(".")[1]),"micro":0}


def test_drift_raises_four_specific_gates_when_old_binary_missing(tmp_path):
    venv,runtime=fixture(tmp_path,has_original=False)
    d=a.audit(venv,runtime=runtime)
    assert d["status"]=="RUNTIME_RESTORE_HOLD"
    assert d["created_python_version"]=="3.12.3"
    assert d["active_python_version"]=="3.14.0"
    assert d["prior_site_metadata_distribution_count"]==len(a.NEEDED)
    assert sorted(a.NEEDED)==d["required_training_packages_in_old_site"]
    assert d["required_training_packages_in_active_site"]==[]
    assert d["issues"]==[
        "active_environment_missing_required_training_packages",
        "interpreter_minor_version_drift",
        "original_venv_interpreter_missing",
        "training_dependencies_stranded_in_prior_site_packages",
    ]
    assert not d["may_train"] and not d["may_resume"]


def test_version_drift_detected_even_when_no_old_packages(tmp_path):
    venv,runtime=fixture(tmp_path)
    old=venv/"lib/python3.12/site-packages"
    for p in list(old.iterdir()):
        p.rmdir()
    d=a.audit(venv,runtime=runtime)
    assert "interpreter_minor_version_drift" in d["issues"]
    assert "training_dependencies_stranded_in_prior_site_packages" not in d["issues"]


def test_matching_interpreter_and_site_is_not_training_permission(tmp_path):
    venv,runtime=fixture(tmp_path,current="3.12")
    d=a.audit(venv,runtime=runtime)
    assert d["issues"]==[]
    assert d["interpreter_and_site_structure_compatible"]
    assert not d["may_train"] and not d["trainer_restore_performed"]


def test_missing_one_live_dependency_fails_closed(tmp_path):
    venv,runtime=fixture(tmp_path,current="3.12")
    (venv/"lib/python3.12/site-packages/peft").rmdir()
    d=a.audit(venv,runtime=runtime)
    assert "active_environment_missing_required_training_packages" in d["issues"]
    assert not d["interpreter_and_site_structure_compatible"]


def test_malformed_venv_config_is_not_accepted(tmp_path):
    venv,runtime=fixture(tmp_path)
    (venv/"pyvenv.cfg").write_text("version = not-a-version\n")
    d=a.audit(venv,runtime=runtime)
    assert "venv_creation_interpreter_unverified" in d["issues"]
    assert "original_venv_interpreter_missing" in d["issues"]


def test_symlink_venv_config_denied(tmp_path):
    venv,runtime=fixture(tmp_path)
    real=tmp_path/"real-config"
    config=venv/"pyvenv.cfg"
    config.rename(real)
    config.symlink_to(real)
    d=a.audit(venv,runtime=runtime)
    assert "venv_creation_interpreter_unverified" in d["issues"]


def test_malicious_symlinked_package_directory_not_credited(tmp_path):
    venv,runtime=fixture(tmp_path,current="3.12")
    target=venv/"lib/python3.12/site-packages/peft"
    moved=tmp_path/"moved-peft"
    target.rename(moved)
    target.symlink_to(moved,target_is_directory=True)
    d=a.audit(venv,runtime=runtime)
    assert "active_environment_missing_required_training_packages" in d["issues"]


def test_missing_interpreter_reported(tmp_path):
    venv,_=fixture(tmp_path)
    d=a.audit(venv)
    assert "active_interpreter_unavailable" in d["issues"]
    assert d["active_python_version"] is None


def test_probe_parses_isolated_python_response(tmp_path):
    venv,_=fixture(tmp_path)
    exe=venv/"bin/python"
    exe.symlink_to(sys.executable)
    value=a.probe_interpreter(exe)
    assert value["major"]==sys.version_info.major
    assert value["minor"]==sys.version_info.minor


def test_duplicate_config_keys_do_not_override_initial_version(tmp_path):
    venv,runtime=fixture(tmp_path)
    with (venv/"pyvenv.cfg").open("a") as f:
        f.write("version = 3.14.4\n")
    d=a.audit(venv,runtime=runtime)
    assert d["created_python_version"]=="3.12.3"
    assert "interpreter_minor_version_drift" in d["issues"]


def test_oversize_config_denied(tmp_path):
    venv,runtime=fixture(tmp_path)
    (venv/"pyvenv.cfg").write_text("version = 3.14.4\n"+"x"*a.CFG_MAX_BYTES)
    d=a.audit(venv,runtime=runtime)
    assert "venv_creation_interpreter_unverified" in d["issues"]


def test_metadata_inspection_is_read_only(tmp_path):
    venv,runtime=fixture(tmp_path)
    before={str(p):p.stat().st_mtime_ns for p in venv.rglob("*") if p.is_file()}
    a.audit(venv,runtime=runtime)
    after={str(p):p.stat().st_mtime_ns for p in venv.rglob("*") if p.is_file()}
    assert before==after


def test_cli_always_hold_even_when_test_mocks_good_state(tmp_path,monkeypatch,capsys):
    venv,runtime=fixture(tmp_path,current="3.12")
    monkeypatch.setattr(a,"VENV",venv)
    monkeypatch.setattr(a,"probe_interpreter",lambda _:runtime)
    monkeypatch.setattr(sys,"argv",["qwen_python_runtime_audit.py","--venv",str(venv)])
    assert a.main()==78
    d=json.loads(capsys.readouterr().out)
    assert d["status"]=="RUNTIME_RESTORE_HOLD"
    assert not d["may_create_or_modify_environment"]
    assert not d["may_train"]

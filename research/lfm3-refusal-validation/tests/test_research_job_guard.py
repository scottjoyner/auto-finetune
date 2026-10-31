"""Synthetic-only capacity safety tests; no checkpoints or model execution."""
from types import SimpleNamespace
import sys

import pytest

from src.research_job_guard import (
    ResearchWriteBlocked, existing_directory, require_research_write_capacity,
)
from src.xstest_450_eval import main as xstest_main


def status(allowed, available=4.0):
    return {
        "large_write_allowed": allowed,
        "available_gib": available,
        "required_available_gib": 20.0,
    }


def test_guard_blocks_before_any_research_output(tmp_path, monkeypatch):
    monkeypatch.setattr("src.research_job_guard.preflight",
                        lambda *args: status(False))
    with pytest.raises(ResearchWriteBlocked, match="BLOCKED"):
        require_research_write_capacity(tmp_path / "not-created" / "output",
                                        local_root=tmp_path)
    assert not (tmp_path / "not-created").exists()


def test_guard_allows_only_measured_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr("src.research_job_guard.preflight",
                        lambda *args: status(True, 40))
    result=require_research_write_capacity(tmp_path / "future", local_root=tmp_path)
    assert result["model_write_allowed"] is True
    assert result["space_reserved"] is False
    assert result["other_processes_blocked"] is False


def test_guard_rejects_unsafe_path_and_capacity(tmp_path):
    bad=tmp_path / "ordinary_file"
    bad.write_text("x")
    with pytest.raises(ValueError):
        existing_directory(bad)
    link=tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        existing_directory(link)
    with pytest.raises(ValueError):
        require_research_write_capacity(tmp_path, local_root=tmp_path,
                                        planned_write_gib=-1)


def test_xstest_main_blocks_before_source_read(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv",
                        ["xstest_450_eval", "--results-dir", str(tmp_path/"none")])
    monkeypatch.setattr("src.xstest_450_eval.require_research_write_capacity",
                        lambda *_: (_ for _ in ()).throw(
                            ResearchWriteBlocked("intentionally blocked")))
    monkeypatch.setattr("src.xstest_450_eval.read_verified_dataset",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("benchmark source should not be opened")))
    with pytest.raises(ResearchWriteBlocked, match="intentionally blocked"):
        xstest_main()
    assert not (tmp_path/"none").exists()


@pytest.mark.parametrize("module_name,output_flag", [
    ("src.arc_capability_eval", "--output"),
    ("src.arc_full_likelihood_eval", "--results-dir"),
    ("src.three_checkpoint_eval", "--results"),
])
def test_other_evaluation_entrypoints_refuse_before_model_work(
        monkeypatch, tmp_path, module_name, output_flag):
    from importlib import import_module
    module=import_module(module_name)
    monkeypatch.setattr(sys, "argv",
                        [module_name, output_flag, str(tmp_path/"out")])
    monkeypatch.setattr(module, "require_research_write_capacity",
                        lambda *_: (_ for _ in ()).throw(
                            ResearchWriteBlocked("disk block")))
    with pytest.raises(ResearchWriteBlocked, match="disk block"):
        module.main()
    assert not (tmp_path/"out").exists()

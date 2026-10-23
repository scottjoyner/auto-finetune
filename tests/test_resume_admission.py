"""Pure resume custody-gate tests. No optimizer deserialization or model/GPU use."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import resume_admission as r


def inputs(tmp_path):
    run = tmp_path / "outputs" / "checkpoints" / "qwen3-8b-sft-r1"
    checkpoint = run / "checkpoint-150"
    checkpoint.mkdir(parents=True)
    clean = tmp_path / "data" / "clean-train.jsonl"
    clean.parent.mkdir()
    clean.write_text("{}\n")
    split = tmp_path / "partition.json"
    split.write_text(json.dumps({
        "train_path": str(clean),
        "contamination": {"status": "clean", "overlap_count": 0}
    }))
    kw = {
        "requested": str(checkpoint),
        "output_dir": str(run),
        "data_path": str(clean),
        "model_name": str(r.EXPECTED_MODEL),
        "backup": {
            "restore_test_passed": True,
            "status": "VERIFIED_RESTORE",
            "may_release_source": False
        },
        "preflight": {
            "restore_test_passed": True,
            "may_resume": True,
            "reasons": [],
            "checkpoint": {"step": 150}
        },
        "run_dir": run,
        "split": split
    }
    return kw


def test_only_fully_verified_synthetic_metadata_passes(tmp_path):
    kw = inputs(tmp_path)
    decision = r.evaluate(**kw)
    assert decision.allowed is True
    assert decision.reason == "independently_verified_resume"


@pytest.mark.parametrize("field,value,reason", [
    ("requested", "", "explicit_absolute_checkpoint_required"),
    ("requested", "checkpoint-150", "explicit_absolute_checkpoint_required"),
    ("requested", "/tmp/checkpoint-foo", "invalid_numeric_checkpoint"),
    ("requested", "/tmp/checkpoint-0", "invalid_numeric_checkpoint"),
    ("model_name", "/dev/null", "base_model_mismatch"),
    ("data_path", "/tmp/other-train.jsonl", "exact_training_split_mismatch"),
    ("output_dir", "/tmp/output", "checkpoint_output_mismatch"),
])
def test_invalid_request_paths_or_inputs_denied(tmp_path, field, value, reason):
    kw = inputs(tmp_path)
    kw[field] = value
    decision = r.evaluate(**kw)
    assert not decision.allowed
    assert decision.reason == reason


def test_checkpoint_outside_exact_parent_denied(tmp_path):
    kw = inputs(tmp_path)
    kw["requested"] = str(tmp_path / "other" / "checkpoint-150")
    assert r.evaluate(**kw).reason == "checkpoint_output_mismatch"


def test_symlinked_checkpoint_denied(tmp_path):
    kw = inputs(tmp_path)
    real = Path(kw["requested"])
    renamed = real.with_name("old-real-checkpoint")
    real.rename(renamed)
    real.symlink_to(renamed, target_is_directory=True)
    assert r.evaluate(**kw).reason == "checkpoint_unavailable_or_symlinked"


def test_missing_backup_or_disclaimer_is_not_restore_custody(tmp_path):
    kw = inputs(tmp_path)
    for metadata in (
        {},
        {"restore_test_passed": False, "status": "VERIFIED_RESTORE", "may_release_source": False},
        {"restore_test_passed": True, "status": "NO_VERIFIED_RESTORE", "may_release_source": False},
        {"restore_test_passed": True, "status": "VERIFIED_RESTORE", "may_release_source": True},
    ):
        kw["backup"] = metadata
        assert r.evaluate(**kw).reason == "restore_custody_not_verified"


def test_preflight_requires_matching_restored_checkpoint_step(tmp_path):
    kw = inputs(tmp_path)
    kw["preflight"]["checkpoint"]["step"] = 100
    assert r.evaluate(**kw).reason == "verified_checkpoint_step_mismatch"


def test_preflight_requires_unambiguous_live_clearance(tmp_path):
    kw = inputs(tmp_path)
    for candidate in (
        {"restore_test_passed": False, "may_resume": True, "reasons": []},
        {"restore_test_passed": True, "may_resume": False, "reasons": []},
        {"restore_test_passed": True, "may_resume": True, "reasons": ["gpu_not_ready"]},
    ):
        kw["preflight"] = candidate
        assert r.evaluate(**kw).reason == "resume_dependency_not_verified"


def test_contaminated_or_missing_manifest_denied(tmp_path):
    kw = inputs(tmp_path)
    split = kw["split"]
    split.write_text(json.dumps({
        "train_path": kw["data_path"],
        "contamination": {"status": "contaminated", "overlap_count": 1}
    }))
    assert r.evaluate(**kw).reason == "training_split_contaminated"
    split.unlink()
    assert r.evaluate(**kw).reason == "split_manifest_absent"


def test_unverified_preflight_never_recovers_by_env_or_path(tmp_path):
    kw = inputs(tmp_path)
    kw["backup"] = {"status": "NO_VERIFIED_RESTORE", "restore_test_passed": False,
                     "may_release_source": False}
    assert not r.evaluate(**kw).allowed


def test_live_uses_both_independent_auditors_and_stays_denied(tmp_path, monkeypatch):
    kw = inputs(tmp_path)
    import scripts.resume_backup_audit as ba
    import scripts.qwen_resume_preflight as qp
    monkeypatch.setattr(ba, "audit", lambda *args: {
        "status": "NO_VERIFIED_RESTORE", "restore_test_passed": False,
        "may_release_source": False
    })
    monkeypatch.setattr(qp, "audit", lambda *args, **kwargs: {
        "restore_test_passed": False, "may_resume": False, "reasons": ["no_restore"],
        "checkpoint": {"step": 150}
    })
    monkeypatch.setattr(r, "RUN", kw["run_dir"])
    monkeypatch.setattr(r, "SPLIT", kw["split"])
    # check_live must honor injected run roots when running fixture tests.
    monkeypatch.setattr(r, "evaluate", lambda requested, **opts: r.Decision(
        False, "restore_custody_not_verified"))
    value = r.check_live(kw["requested"], output_dir=kw["output_dir"],
                         data_path=kw["data_path"], model_name=kw["model_name"])
    assert value.allowed is False
    assert value.reason == "restore_custody_not_verified"


def test_invalid_preflight_checkpoint_shape_denies_not_raises(tmp_path):
    kw = inputs(tmp_path)
    for unexpected in (None, "checkpoint-150", [], 150):
        kw["preflight"] = {
            "restore_test_passed": True, "may_resume": True,
            "reasons": [], "checkpoint": unexpected,
        }
        result = r.evaluate(**kw)
        assert result.allowed is False
        assert result.reason == "verified_checkpoint_step_mismatch"


def test_live_denial_does_not_depend_on_fake_allow_metadata(tmp_path, monkeypatch):
    import scripts.resume_backup_audit as ba
    import scripts.qwen_resume_preflight as qp
    kw = inputs(tmp_path)
    monkeypatch.setattr(ba, "audit", lambda *args: {
        "restore_test_passed": False, "status": "NO_VERIFIED_RESTORE",
        "may_release_source": False
    })
    monkeypatch.setattr(qp, "audit", lambda *args, **kwargs: {
        "restore_test_passed": False, "may_resume": False,
        "reasons": ["training_code_has_no_resume_entrypoint"],
        "checkpoint": {"step": 150}
    })
    original_evaluate = r.evaluate
    monkeypatch.setattr(r, "evaluate", lambda requested, **arguments: original_evaluate(
        requested, run_dir=kw["run_dir"], split=kw["split"], **arguments))
    decision = r.check_live(kw["requested"], output_dir=kw["output_dir"],
                            data_path=kw["data_path"], model_name=kw["model_name"])
    assert decision.reason == "restore_custody_not_verified"
    assert decision.allowed is False


def test_report_cannot_grant_training_even_if_checker_is_mocked_allowed(monkeypatch, capsys):
    import sys
    from scripts import resume_admission_report as report
    monkeypatch.setattr(r, "check_live", lambda *args, **kwargs: r.Decision(True, "synthetic"))
    monkeypatch.setattr(sys, "argv", [
        "resume_admission_report.py",
        "--checkpoint", "/safe/checkpoint-150",
        "--output-dir", "/safe",
        "--dataset", "/clean.jsonl",
        "--model", str(r.EXPECTED_MODEL),
    ])
    assert report.main() == 78
    data = json.loads(capsys.readouterr().out)
    assert data["may_train"] is False
    assert data["may_resume"] is False
    assert data["restore_test_passed"] is False

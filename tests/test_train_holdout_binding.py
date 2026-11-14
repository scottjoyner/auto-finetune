"""Synthetic direct-trainer benchmark provenance admission boundary tests.

These tests never load a model or acquire GPU resources. Direct trainer
admission and benchmark checks are replaced by explicit synthetic decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import train
from src import training_admission as admission
from src import quality_bundle as qb


class Cfg:
    def __init__(self, path):
        self.root = path
    def path(self, name):
        assert name == "dataset_dir"
        return str(self.root)
    def get(self, section, key, default=None):
        return "synthetic/no-training-model" if (section, key) == ("train", "model_name") else default


@pytest.mark.parametrize(("status", "code"), [
    ("unverifiable", 78),
    ("inactive", 78),
    ("skipped", 78),
    ("missing", 78),
    ("contaminated", 3),
])
def test_real_train_never_accepts_unsupported_holdout_receipt(
        tmp_path, monkeypatch, status, code):
    monkeypatch.setenv("TRAIN_ALLOW_UNVERIFIED", "1")
    monkeypatch.setenv("TRAIN_SKIP_HOLDOUT_CHECK", "1")
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(train, "_reject_held_out", lambda path: status)
    monkeypatch.setattr(
        train, "validate_dataset",
        lambda *a: pytest.fail("dataset scanned after unknown benchmark receipt"),
    )
    assert train.main(Cfg(tmp_path)) == code


def test_holdout_check_exception_denies_real_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(
        train, "_reject_held_out",
        lambda *a: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(
        train, "validate_dataset",
        lambda *a: pytest.fail("dataset loaded after provenance failure"),
    )
    assert train.main(Cfg(tmp_path)) == 78


def test_verified_bundle_has_correct_native_trainer_sidecar(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    data = dataset / qb.DATA_NAME
    data.write_text(json.dumps({"messages": [
        {"role": "user", "content": "synthetic example"},
        {"role": "assistant", "content": "synthetic answer"},
    ]}) + "\n")
    data.with_suffix(".provenance.jsonl").write_text(
        json.dumps({"row": 0, "session_id": "fake-clean-session"}) + "\n",
    )
    receipt = qb.publish_quality_bundle(data, dataset)
    verified = qb.bundle_path(dataset, receipt["bundle_id"]) / qb.DATA_NAME
    from src.format_dataset import verify_dataset
    result = verify_dataset(verified, {"synthetic-heldout"})
    assert result["status"] == "clean"
    assert result["n_rows"] == 1
    assert result["n_sessions"] == 1


def test_heldout_id_in_verified_bundle_remains_detectable(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    source = dataset / qb.DATA_NAME
    source.write_text(json.dumps({"messages": [
        {"role": "user", "content": "synthetic"},
        {"role": "assistant", "content": "response"},
    ]}) + "\n")
    source.with_suffix(".provenance.jsonl").write_text(
        json.dumps({"row": 0, "session_id": "fake-heldout"}) + "\n",
    )
    receipt = qb.publish_quality_bundle(source, dataset)
    verified = qb.bundle_path(dataset, receipt["bundle_id"]) / qb.DATA_NAME
    from src.format_dataset import verify_dataset
    result = verify_dataset(verified, {"fake-heldout"})
    assert result["status"] == "contaminated"
    assert result["leaked"][0]["session_id"] == "fake-heldout"


def test_mocked_clean_holdout_status_can_reach_dataset_validation(
        tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(train, "_reject_held_out", lambda *a: "clean")
    def marker(*a):
        raise RuntimeError("validation reached, but model load prohibited")
    monkeypatch.setattr(train, "validate_dataset", marker)
    with pytest.raises(RuntimeError, match="validation reached"):
        train.main(Cfg(tmp_path))


def _fixture_bundle_for_native_trainer(tmp_path):
    root = tmp_path / "input-datasets"
    root.mkdir()
    data = root / qb.DATA_NAME
    data.write_text(json.dumps({"messages": [
        {"role": "user", "content": "synthetic question"},
        {"role": "assistant", "content": "synthetic answer"},
    ]}) + "\n")
    data.with_suffix(".provenance.jsonl").write_text(
        json.dumps({"row": 0, "session_id": "synthetic-non-heldout"}) + "\n"
    )
    receipt = qb.publish_quality_bundle(data, root)
    directory = qb.bundle_path(root, receipt["bundle_id"])
    return root, directory


def test_native_trainer_rechecks_exact_quality_receipt(tmp_path):
    root, directory = _fixture_bundle_for_native_trainer(tmp_path)
    native = directory / qb.DATA_NAME
    assert train._verify_native_quality_bundle(
        str(directory), str(native), source="neo4j-traces", label=None
    )
    assert not train._verify_native_quality_bundle(
        str(directory), str(native), source="opencode", label=None
    )
    assert not train._verify_native_quality_bundle(
        str(directory), str(native), source="neo4j-traces", label="other"
    )
    assert not train._verify_native_quality_bundle(
        str(directory), str(directory / "train.other.jsonl"),
        source="neo4j-traces", label=None
    )


@pytest.mark.parametrize("tamper", [qb.DATA_NAME, qb.PROVENANCE_NAME, qb.RECEIPT_NAME])
def test_native_trainer_denies_changed_bundle_before_data_access(tmp_path, tamper):
    root, directory = _fixture_bundle_for_native_trainer(tmp_path)
    (directory / tamper).write_text("synthetic tampering")
    with pytest.raises(ValueError):
        train._verify_native_quality_bundle(
            str(directory), str(directory / qb.DATA_NAME),
            source="neo4j-traces", label=None
        )


def test_native_trainer_denies_source_changed_after_qa(tmp_path):
    root, directory = _fixture_bundle_for_native_trainer(tmp_path)
    (root / qb.DATA_NAME).write_text("new unverified data")
    with pytest.raises(ValueError):
        train._verify_native_quality_bundle(
            str(directory), str(directory / qb.DATA_NAME),
            source="neo4j-traces", label=None
        )


def test_native_trainer_denies_nested_fake_bundle(tmp_path):
    directory = tmp_path / "quality-bundles" / "unexpected" / ("qb-" + "a" * 64)
    assert not train._verify_native_quality_bundle(
        str(directory), str(directory / qb.DATA_NAME),
        source="neo4j-traces", label=None
    )


def test_direct_training_holds_on_fake_native_quality_directory(
        tmp_path, monkeypatch):
    fake = tmp_path / "quality-bundles" / ("qb-" + "a" * 64)
    monkeypatch.setenv("TRAIN_DATASET_DIR", str(fake))
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(train, "_reject_held_out",
                        lambda *a: pytest.fail("holdout evaluated after missing QA bundle"))
    monkeypatch.setattr(train, "validate_dataset",
                        lambda *a: pytest.fail("model dataset loaded without QA"))
    assert train.main(Cfg(tmp_path), source="neo4j-traces") == 78


def test_direct_training_holds_on_wrong_native_target(
        tmp_path, monkeypatch):
    root, bundle = _fixture_bundle_for_native_trainer(tmp_path)
    monkeypatch.setenv("TRAIN_DATASET_DIR", str(bundle))
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(train, "_reject_held_out",
                        lambda *a: pytest.fail("unapproved source reached holdout"))
    assert train.main(Cfg(tmp_path), source="wrong-label") == 78


def test_native_bundle_rechecked_after_dataset_read(tmp_path, monkeypatch):
    root, bundle = _fixture_bundle_for_native_trainer(tmp_path)
    monkeypatch.setenv("TRAIN_DATASET_DIR", str(bundle))
    monkeypatch.setenv("TRAIN_MODEL_NAME", "synthetic/no-training-model")
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic-admitted"))
    monkeypatch.setattr(train, "_reject_held_out", lambda *a: "clean")
    calls = []
    def mutate_after_read(path):
        calls.append(path)
        (bundle / qb.DATA_NAME).write_text("changed-during-read")
        return [{"messages": [{"role": "user", "content": "synthetic"}]}]
    monkeypatch.setattr(train, "validate_dataset", mutate_after_read)
    monkeypatch.setattr(train, "_resolve_backend",
                        lambda *a: pytest.fail("model backend reached after QA mutation"))
    assert train.main(Cfg(tmp_path), source="neo4j-traces") == 78
    assert len(calls) == 1

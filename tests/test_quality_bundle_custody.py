"""Temporary, exact-source QA bundle custody, integrity and no-clobber tests.

This module never uses production datasets, credentials, inference or GPUs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import quality_bundle as qb


def record(sid, content="synthetic answer"):
    return {
        "messages": [
            {"role": "user", "content": f"synthetic question {sid}"},
            {"role": "assistant", "content": content},
        ]
    }


def setup(tmp_path, records=None, sids=None):
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    records = records if records is not None else [record("a"), record("b")]
    sids = sids if sids is not None else [f"fake-{i}" for i in range(len(records))]
    assert len(records) == len(sids)
    source = dataset / qb.DATA_NAME
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    sidecar = source.with_suffix(".provenance.jsonl")
    sidecar.write_text("".join(json.dumps({"row": n, "session_id": sid}) + "\n"
                               for n, sid in enumerate(sids)))
    return dataset, source, sidecar


def test_bundle_source_and_sidecar_are_required(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    sidecar.unlink()
    with pytest.raises(ValueError, match="quality_source_or_provenance_missing"):
        qb.publish_quality_bundle(source, dataset)
    assert not (dataset / "quality-bundles").exists()


def test_bundle_rejects_missing_source_without_writing(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match="quality_source_or_provenance_missing"):
        qb.publish_quality_bundle(source, dataset)


@pytest.mark.parametrize("corrupt", [
    "not-json\n",
    '["bad-type"]\n',
    '{"not":"messages"}\n',
    '{"messages":[]}\n',
    '{"messages":["bad"]}\n',
])
def test_corrupt_training_jsonl_is_hold(tmp_path, corrupt):
    dataset, source, _ = setup(tmp_path)
    source.write_text(corrupt)
    with pytest.raises(ValueError):
        qb.publish_quality_bundle(source, dataset)


@pytest.mark.parametrize("corrupt", [
    "bad\n",
    '{"row":0,"session_id":""}\n{"row":1,"session_id":"fake"}\n',
    '{"row":0,"session_id":"fake"}\n',
    '{"row":1,"session_id":"fake"}\n{"row":0,"session_id":"fake2"}\n',
    '{"row":false,"session_id":"fake"}\n{"row":1,"session_id":"fake2"}\n',
])
def test_unverifiable_row_provenance_holds(tmp_path, corrupt):
    dataset, source, sidecar = setup(tmp_path)
    sidecar.write_text(corrupt)
    with pytest.raises(ValueError):
        qb.publish_quality_bundle(source, dataset)


def test_immutable_bundle_uses_exact_trainer_filename_and_provenance(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    receipt = qb.publish_quality_bundle(source, dataset)
    bundle = qb.bundle_path(dataset, receipt["bundle_id"])
    assert bundle.is_dir()
    assert sorted(p.name for p in bundle.iterdir()) == sorted([
        qb.DATA_NAME, qb.PROVENANCE_NAME, qb.RECEIPT_NAME
    ])
    assert len((bundle / qb.DATA_NAME).read_text().splitlines()) == 2
    assert len((bundle / qb.PROVENANCE_NAME).read_text().splitlines()) == 2
    assert qb.verify_quality_bundle(dataset, receipt["bundle_id"], source) == receipt
    assert receipt["source_rows"] == receipt["accepted_rows"] == 2


def test_filter_preserves_correct_original_session_ids(tmp_path):
    records = [
        record("one"), {"messages": []}, record("three"),
    ]
    dataset, source, _ = setup(tmp_path, records=records,
                               sids=["synthetic-A", "synthetic-EXCLUDED", "synthetic-C"])
    receipt = qb.publish_quality_bundle(source, dataset)
    base = qb.bundle_path(dataset, receipt["bundle_id"])
    kept = [json.loads(line) for line in (base / qb.PROVENANCE_NAME).read_text().splitlines()]
    assert kept == [
        {"row": 0, "session_id": "synthetic-A"},
        {"row": 1, "session_id": "synthetic-C"},
    ]
    assert receipt["source_rows"] == 3
    assert receipt["accepted_rows"] == 2


def test_majority_poor_records_are_hold(tmp_path):
    records = [record("keep"), {"messages": []}, {"messages": []}]
    dataset, source, _ = setup(tmp_path, records=records)
    with pytest.raises(ValueError, match="quality_failure_rate"):
        qb.publish_quality_bundle(source, dataset)
    assert not (dataset / "quality-bundles").exists()


def test_all_poor_records_hold_before_any_directory(tmp_path):
    dataset, source, _ = setup(tmp_path, records=[{"messages": []}])
    with pytest.raises(ValueError, match="quality_no_passing_records"):
        qb.publish_quality_bundle(source, dataset)


def test_exact_repeat_reuses_identical_immutable_bundle(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    first = qb.publish_quality_bundle(source, dataset)
    original = (qb.bundle_path(dataset, first["bundle_id"]) / qb.DATA_NAME).read_bytes()
    second = qb.publish_quality_bundle(source, dataset)
    assert second == first
    assert (qb.bundle_path(dataset, first["bundle_id"]) / qb.DATA_NAME).read_bytes() == original
    assert len(list((dataset / "quality-bundles").glob("qb-*"))) == 1


def test_changed_source_produces_different_bundle_not_overwrite(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    first = qb.publish_quality_bundle(source, dataset)
    source.write_text(source.read_text() + json.dumps(record("new")) + "\n")
    sidecar.write_text(sidecar.read_text() + json.dumps({
        "row": 2, "session_id": "synthetic-new"
    }) + "\n")
    second = qb.publish_quality_bundle(source, dataset)
    assert second["bundle_id"] != first["bundle_id"]
    assert qb.bundle_path(dataset, first["bundle_id"]).exists()
    assert qb.bundle_path(dataset, second["bundle_id"]).exists()
    with pytest.raises(ValueError, match="quality_receipt_not_current"):
        qb.verify_quality_bundle(dataset, first["bundle_id"], source)


@pytest.mark.parametrize("tamper", [qb.DATA_NAME, qb.PROVENANCE_NAME, qb.RECEIPT_NAME])
def test_any_tampered_bundle_file_is_hold(tmp_path, tamper):
    dataset, source, _ = setup(tmp_path)
    receipt = qb.publish_quality_bundle(source, dataset)
    path = qb.bundle_path(dataset, receipt["bundle_id"]) / tamper
    path.write_text("ALTERED")
    with pytest.raises(ValueError):
        qb.verify_quality_bundle(dataset, receipt["bundle_id"], source)
    with pytest.raises(ValueError):
        qb.publish_quality_bundle(source, dataset)


def test_missing_receipt_from_interrupted_publish_is_hold(tmp_path):
    dataset, source, _ = setup(tmp_path)
    receipt = qb.publish_quality_bundle(source, dataset)
    (qb.bundle_path(dataset, receipt["bundle_id"]) / qb.RECEIPT_NAME).unlink()
    with pytest.raises(ValueError, match="quality_bundle_incomplete"):
        qb.publish_quality_bundle(source, dataset)


def test_provenance_change_invalidates_earlier_receipt(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    receipt = qb.publish_quality_bundle(source, dataset)
    sidecar.write_text(sidecar.read_text().replace("fake-0", "tampered"))
    with pytest.raises(ValueError, match="quality_receipt_not_current"):
        qb.verify_quality_bundle(dataset, receipt["bundle_id"], source)


@pytest.mark.parametrize("threshold", [-1, .5, 1.1, float("nan")])
def test_unapproved_threshold_is_hold(tmp_path, threshold):
    dataset, source, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        qb.publish_quality_bundle(source, dataset, min_score=threshold)


def test_bundle_id_path_traversal_denied(tmp_path):
    for value in ("../../secret", "qb-unknown", "qb-" + "g" * 64, ""):
        with pytest.raises(ValueError, match="quality_bundle_id_invalid"):
            qb.bundle_path(tmp_path, value)


def test_source_sidecar_symlinks_not_approved(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    other = dataset / "sidecar-real"
    sidecar.rename(other)
    sidecar.symlink_to(other)
    with pytest.raises(ValueError, match="quality_source_or_provenance_missing"):
        qb.publish_quality_bundle(source, dataset)


def test_receipt_publication_failure_leaves_unverifiable_bundle(tmp_path, monkeypatch):
    dataset, source, _ = setup(tmp_path)
    original = qb.os.link
    def fake_link(a, b):
        if Path(b).name == qb.RECEIPT_NAME:
            raise OSError("synthetic interruption")
        return original(a, b)
    monkeypatch.setattr(qb.os, "link", fake_link)
    with pytest.raises(OSError):
        qb.publish_quality_bundle(source, dataset)
    bundles = list((dataset / "quality-bundles").glob("qb-*"))
    assert len(bundles) == 1
    assert not (bundles[0] / qb.RECEIPT_NAME).exists()
    assert not list((dataset / "quality-bundles").glob(".qb-stage-*"))
    with pytest.raises(ValueError, match="quality_bundle_incomplete"):
        qb.publish_quality_bundle(source, dataset)


def test_symlinked_quality_bundle_parent_denies_reuse(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    receipt = qb.publish_quality_bundle(source, dataset)
    parent = dataset / "quality-bundles"
    moved = dataset / "original-private-bundles"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="quality_bundle_missing"):
        qb.verify_quality_bundle(dataset, receipt["bundle_id"], source)
    with pytest.raises(ValueError, match="quality_bundle_parent_symlink"):
        qb.publish_quality_bundle(source, dataset)


def test_symlink_parent_denies_initial_quality_publication(tmp_path):
    dataset, source, sidecar = setup(tmp_path)
    outside = tmp_path / "unrelated-directory"
    outside.mkdir()
    (dataset / "quality-bundles").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="quality_bundle_parent_symlink"):
        qb.publish_quality_bundle(source, dataset)
    assert list(outside.iterdir()) == []

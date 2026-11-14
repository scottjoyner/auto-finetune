"""Offline, immutable quality/provenance bundle for realtime training.

No model, network or credential access. A bundle becomes valid only when its
data, filtered row-aligned provenance and receipt all verify. An interrupted
write or changed upstream source fails closed rather than replacing prior data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

from src.data_quality import score_session

SCHEMA = "auto-finetune.quality-bundle.v1"
DATA_NAME = "train.neo4j-traces.jsonl"
PROVENANCE_NAME = "train.neo4j-traces.provenance.jsonl"
RECEIPT_NAME = "receipt.json"
ID_RE = re.compile(r"qb-[0-9a-f]{64}")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_bytes(source_path: Path) -> tuple[bytes, bytes]:
    sidecar = source_path.with_suffix(".provenance.jsonl")
    if (source_path.is_symlink() or sidecar.is_symlink()
            or not source_path.is_file() or not sidecar.is_file()):
        raise ValueError("quality_source_or_provenance_missing")
    return source_path.read_bytes(), sidecar.read_bytes()


def _decode_lines(blob: bytes) -> list[dict]:
    records = []
    try:
        for line in blob.splitlines():
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("quality_record_not_object")
                records.append(record)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError("quality_input_malformed") from exc
    if not records:
        raise ValueError("quality_input_empty")
    return records


def _checked_records(data: bytes, provenance: bytes) -> tuple[list[dict], list[dict]]:
    records = _decode_lines(data)
    sidecar = _decode_lines(provenance)
    if len(sidecar) != len(records):
        raise ValueError("quality_provenance_count_mismatch")
    for index, (record, row) in enumerate(zip(records, sidecar)):
        if (not isinstance(record.get("messages"), list)
                or not all(isinstance(message, dict) for message in record["messages"])
                or type(row.get("row")) is not int or row["row"] != index
                or not isinstance(row.get("session_id"), str)
                or not row["session_id"]):
            raise ValueError("quality_provenance_invalid")
    return records, sidecar


def _encode(records: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
         + "\n").encode("utf-8")
        for row in records
    )


def _bundle_id(source_data: bytes, source_provenance: bytes, min_score: float) -> str:
    identity = (
        SCHEMA + "\n" + _digest(source_data) + "\n" +
        _digest(source_provenance) + "\n" + str(min_score)
    ).encode()
    return "qb-" + _digest(identity)


def _content(source_data: bytes, source_provenance: bytes,
             min_score: float) -> tuple[bytes, bytes, dict]:
    records, sidecar = _checked_records(source_data, source_provenance)
    chosen = []
    mapped = []
    for index, record in enumerate(records):
        quality = score_session(record)
        if (not math.isfinite(quality.overall)
                or not 0 <= quality.overall <= 1):
            raise ValueError("quality_score_invalid")
        if quality.passed and quality.overall >= min_score:
            chosen.append(record)
            mapped.append({
                "row": len(chosen) - 1,
                "session_id": sidecar[index]["session_id"],
            })
    if not chosen:
        raise ValueError("quality_no_passing_records")
    if len(records) - len(chosen) > len(records) / 2:
        raise ValueError("quality_failure_rate")
    data = _encode(chosen)
    provenance = _encode(mapped)
    identifier = _bundle_id(source_data, source_provenance, min_score)
    receipt = {
        "schema": SCHEMA,
        "bundle_id": identifier,
        "source_sha256": _digest(source_data),
        "source_provenance_sha256": _digest(source_provenance),
        "dataset_sha256": _digest(data),
        "provenance_sha256": _digest(provenance),
        "source_rows": len(records),
        "accepted_rows": len(chosen),
        "min_score": min_score,
    }
    return data, provenance, receipt


def bundle_path(dataset_dir: str | Path, bundle_id: str) -> Path:
    if not ID_RE.fullmatch(bundle_id):
        raise ValueError("quality_bundle_id_invalid")
    return Path(dataset_dir) / "quality-bundles" / bundle_id


def verify_quality_bundle(dataset_dir: str | Path, bundle_id: str,
                          source_path: str | Path) -> dict:
    """Reverify source, provenance, and receipt at each training boundary."""
    base = bundle_path(dataset_dir, bundle_id)
    if base.is_symlink() or not base.is_dir():
        raise ValueError("quality_bundle_missing")
    paths = [base / name for name in (DATA_NAME, PROVENANCE_NAME, RECEIPT_NAME)]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("quality_bundle_incomplete")
    try:
        receipt = json.loads(paths[2].read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("quality_receipt_invalid") from exc
    if (not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA
            or receipt.get("bundle_id") != bundle_id
            or receipt.get("min_score") != 0.6):
        raise ValueError("quality_receipt_invalid")
    source, source_provenance = _source_bytes(Path(source_path))
    data, provenance, expected = _content(source, source_provenance, 0.6)
    if receipt != expected:
        raise ValueError("quality_receipt_not_current")
    if paths[0].read_bytes() != data or paths[1].read_bytes() != provenance:
        raise ValueError("quality_bundle_modified")
    if receipt["bundle_id"] != _bundle_id(source, source_provenance, 0.6):
        raise ValueError("quality_bundle_identity_mismatch")
    return receipt


def publish_quality_bundle(source_path: str | Path,
                           dataset_dir: str | Path,
                           min_score: float = 0.6) -> dict:
    """Publish a vetted dataset and aligned provenance without overwrites.

    A repeated run with identical source and provenance reuses the *verified*
    immutable bundle. Changed source is given a different bundle ID. An
    incomplete/conflicting bundle is never silently repaired or overwritten.
    """
    if (isinstance(min_score, bool) or not isinstance(min_score, (float, int))
            or not math.isfinite(min_score) or not 0 <= min_score <= 1):
        raise ValueError("quality_threshold_invalid")
    if min_score != 0.6:
        # The protected trainer-bound contract pins its exact quality policy.
        raise ValueError("quality_policy_not_approved")
    source_path = Path(source_path)
    source, source_provenance = _source_bytes(source_path)
    data, provenance, receipt = _content(source, source_provenance, min_score)
    target = bundle_path(dataset_dir, receipt["bundle_id"])
    parent = target.parent
    if target.exists() or target.is_symlink():
        verify_quality_bundle(dataset_dir, receipt["bundle_id"], source_path)
        return receipt
    # No publishing stale snapshot after source or sidecar changed in-flight.
    if _source_bytes(source_path) != (source, source_provenance):
        raise ValueError("quality_source_changed_during_check")

    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".qb-stage-", dir=parent))
    try:
        for name, blob in (
            (DATA_NAME, data),
            (PROVENANCE_NAME, provenance),
            (RECEIPT_NAME, (_encode([receipt]))),
        ):
            with (stage / name).open("wb") as output:
                output.write(blob)
                output.flush()
                os.fsync(output.fileno())
        # mkdir is atomic and refuses to replace an existing directory.
        target.mkdir(exist_ok=False)
        # Receipt is linked LAST. An interrupted publication cannot pass
        # verification, and subsequent retries must HOLD for manual review.
        for name in (DATA_NAME, PROVENANCE_NAME, RECEIPT_NAME):
            os.link(stage / name, target / name)
        verify_quality_bundle(dataset_dir, receipt["bundle_id"], source_path)
        return receipt
    finally:
        shutil.rmtree(stage)

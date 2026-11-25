"""Ephemeral test keys and 5 tiny synthetic files; no production model access."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src import lfm3_artifact_custody as custody
from src import lfm3_release_receipt_verifier as verifier
from src import model_export

NOW = datetime(2026, 10, 29, 5, 0, tzinfo=timezone.utc)
KEY_ID = "ephemeral-test"


@pytest.fixture
def keys():
    signing = Ed25519PrivateKey.generate()
    return signing, {KEY_ID: signing.public_key()}


@pytest.fixture
def records(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    files = {}
    blob_map = {}
    for index, field in enumerate(sorted(custody.DIGEST_FIELDS)):
        name = f"synthetic-{index}.bin"
        data = (field.encode() + b" offline-only ") * (index + 1)
        (root / name).write_bytes(data)
        files[field] = name
        blob_map[field] = hashlib.sha256(data).hexdigest()
    request = {
        "base_model_repo": "LiquidAI/LFM3-1.7B",
        "base_model_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "output_format": "gguf",
        "quantization": "q8_0",
        "output_name": "approved-poc.gguf",
        **blob_map,
    }
    return root, files, request


def signed(signing, request, *, rid="a" * 32):
    manifest = dict(request)
    manifest.update({
        "release_id": rid,
        "issued_at": "2026-10-29T04:59:00Z",
        "expires_at": "2026-10-29T05:30:00Z",
    })
    sig = signing.sign(verifier.signing_bytes(KEY_ID, manifest))
    return json.dumps({
        "schema": verifier.SCHEMA,
        "key_id": KEY_ID,
        "manifest": manifest,
        "signature_b64": base64.b64encode(sig).decode(),
    }, sort_keys=True, separators=(",", ":")).encode()


def check(raw, trust, records, **options):
    root, names, request = records
    kwargs = {
        "trusted_public_keys": trust,
        "expected_request": request,
        "artifact_root": root,
        "filenames_by_digest_field": names,
        "now_utc": NOW,
    }
    kwargs.update(options)
    return custody.check_signed_artifact_bytes(raw, **kwargs)


def test_five_real_local_synthetic_byte_streams_match_without_authority(keys, records):
    signer, trust = keys
    r = check(signed(signer, records[2]), trust, records)
    assert r.status == "local_hashes_match_signed_evidence_non_authorizing"
    assert r.signature_verified
    assert r.five_content_hashes_verified and r.root_descriptor_anchored
    assert not r.heldout_independence_verified
    assert not r.artifact_source_independently_attested
    assert not r.export_permitted and not r.promotion_permitted
    assert not r.fleet_wide_replay_protected


@pytest.mark.parametrize("field", sorted(custody.DIGEST_FIELDS))
def test_any_mutated_artifact_is_rejected(keys, records, field):
    signer, trust = keys
    raw = signed(signer, records[2])
    root, names, request = records
    (root / names[field]).write_bytes(b"CORRUPTED")
    r = check(raw, trust, records)
    assert r.status == "blocked"
    assert r.signature_verified and not r.five_content_hashes_verified
    assert not r.export_permitted


def test_valid_signature_wrong_signer_cannot_read_artifact_bytes(keys, records, monkeypatch):
    signer, trust = keys
    attacker = Ed25519PrivateKey.generate()
    raw = signed(attacker, records[2])
    opened = Mock(side_effect=AssertionError("untrusted file access"))
    with monkeypatch.context() as m:
        m.setattr("os.open", opened)
        result = check(raw, trust, records)
    assert result.status == "blocked"
    assert not result.signature_verified
    opened.assert_not_called()


@pytest.mark.parametrize("field", sorted(custody.DIGEST_FIELDS))
def test_changing_signed_manifest_request_fails_before_open(keys, records, monkeypatch, field):
    signer, trust = keys
    raw = signed(signer, records[2])
    expected = dict(records[2])
    expected[field] = "f" * 64
    opened = Mock(side_effect=AssertionError("artifact IO before signature check"))
    with monkeypatch.context() as m:
        m.setattr("os.open", opened)
        result = check(raw, trust, records, expected_request=expected)
    assert result.status == "blocked"
    opened.assert_not_called()


@pytest.mark.parametrize("field", sorted(custody.DIGEST_FIELDS))
def test_artifact_symlink_rejected(keys, records, field):
    signer, trust = keys
    root, names, request = records
    raw = signed(signer, request)
    link = root / names[field]
    target = root / ("original-" + names[field])
    link.rename(target)
    link.symlink_to(target)
    result = check(raw, trust, records)
    assert result.status == "blocked"
    assert target.read_bytes() != b""


def test_root_symlink_rejected(keys, records, tmp_path):
    signer, trust = keys
    root, names, request = records
    alias = tmp_path / "root-alias"
    alias.symlink_to(root, target_is_directory=True)
    result = check(signed(signer, request), trust, records, artifact_root=alias)
    assert result.status == "blocked"


def test_root_intermediate_symlink_rejected(keys, records, tmp_path):
    signer, trust = keys
    root, names, request = records
    intermediate = tmp_path / "alias-parent"
    intermediate.symlink_to(tmp_path, target_is_directory=True)
    alias = intermediate / root.name
    result = check(signed(signer, request), trust, records, artifact_root=alias)
    assert result.status == "blocked"


def test_private_directory_required(keys, records):
    signer, trust = keys
    root, names, request = records
    root.chmod(0o755)
    assert check(signed(signer, request), trust, records).status == "blocked"


def test_relative_root_rejected(keys, records):
    signer, trust = keys
    raw = signed(signer, records[2])
    assert check(raw, trust, records, artifact_root=Path("relative/root")).status == "blocked"


@pytest.mark.parametrize("bad", [
    "nested/file.bin", "../escape.bin", "..", ".",
    "file..bin", "contains space.bin", "☃.bin", "",
])
def test_bad_filenames_rejected_without_path_walk(keys, records, bad):
    signer, trust = keys
    names = dict(records[1])
    names[next(iter(names))] = bad
    assert check(signed(signer, records[2]), trust, records,
                 filenames_by_digest_field=names).status == "blocked"


def test_duplicate_file_name_rejected(keys, records):
    signer, trust = keys
    names = dict(records[1])
    fields = sorted(names)
    names[fields[1]] = names[fields[0]]
    result = check(signed(signer, records[2]), trust, records,
                   filenames_by_digest_field=names)
    assert result.status == "blocked"


@pytest.mark.parametrize("mapping", [{}, {"extra": "file"}, "invalid", None])
def test_unexpected_mapping_rejected(keys, records, mapping):
    signer, trust = keys
    result = check(signed(signer, records[2]), trust, records,
                   filenames_by_digest_field=mapping)
    assert result.status == "blocked"


def test_hardlinked_file_is_rejected(keys, records):
    signer, trust = keys
    root, names, request = records
    victim = root / names[next(iter(names))]
    os.link(victim, root / "hardlink")
    assert check(signed(signer, request), trust, records).status == "blocked"


def test_fifo_is_rejected_without_hanging(keys, records):
    signer, trust = keys
    root, names, request = records
    victim = root / names[next(iter(names))]
    victim.unlink()
    os.mkfifo(victim)
    result = check(signed(signer, request), trust, records)
    assert result.status == "blocked"


def test_directory_instead_of_file_rejected(keys, records):
    signer, trust = keys
    root, names, request = records
    victim = root / names[next(iter(names))]
    victim.unlink()
    victim.mkdir()
    assert check(signed(signer, request), trust, records).status == "blocked"


def test_missing_file_rejected(keys, records):
    signer, trust = keys
    root, names, request = records
    (root / names[next(iter(names))]).unlink()
    assert check(signed(signer, request), trust, records).status == "blocked"


def test_zero_byte_artifact_matches_only_if_signed_digest_of_empty(keys, records):
    signer, trust = keys
    root, names, request = records
    field = next(iter(names))
    (root / names[field]).write_bytes(b"")
    request[field] = hashlib.sha256(b"").hexdigest()
    assert check(signed(signer, request), trust, records).five_content_hashes_verified


def test_content_change_during_read_is_rejected(keys, records, monkeypatch):
    signer, trust = keys
    raw = signed(signer, records[2])
    root, names, request = records
    target = root / names[sorted(names)[0]]
    real_read = os.read
    changed = False

    def altered_read(fd, n):
        nonlocal changed
        data = real_read(fd, n)
        if not changed and data:
            changed = True
            with target.open("ab") as out:
                out.write(b"MUTATED_WHILE_READING")
        return data

    with monkeypatch.context() as m:
        m.setattr("os.read", altered_read)
        r = check(raw, trust, records)
    assert changed
    assert r.status == "blocked"


def test_content_change_earlier_file_detected_after_later_read(keys, records, monkeypatch):
    signer, trust = keys
    raw = signed(signer, records[2])
    root, names, request = records
    fields = sorted(names)
    first = root / names[fields[0]]
    real_read = os.read
    reads = 0

    def altered_read(fd, n):
        nonlocal reads
        data = real_read(fd, n)
        reads += 1
        if reads == 2:
            with first.open("ab") as out:
                out.write(b"CHANGED_AFTER_HASH")
        return data

    with monkeypatch.context() as m:
        m.setattr("os.read", altered_read)
        result = check(raw, trust, records)
    assert reads >= 2
    assert result.status == "blocked"


def test_oversized_sparse_file_refused_before_read(keys, records):
    signer, trust = keys
    root, names, request = records
    file = root / names[next(iter(names))]
    with file.open("r+b") as f:
        f.truncate(custody.MAX_ARTIFACT_BYTES + 1)
    assert check(signed(signer, request), trust, records).status == "blocked"


def test_existing_files_unchanged_and_no_output_created(keys, records, tmp_path):
    signer, trust = keys
    root, names, request = records
    before = {n: (root / n).read_bytes() for n in names.values()}
    r = check(signed(signer, request), trust, records)
    assert r.five_content_hashes_verified
    after = {n: (root / n).read_bytes() for n in names.values()}
    assert before == after
    assert sorted(x.name for x in root.iterdir()) == sorted(names.values())
    assert not (tmp_path / "model.gguf").exists()


def test_public_result_does_not_disclose_signed_content(keys, records):
    signer, trust = keys
    output = check(signed(signer, records[2]), trust, records).as_public_dict()
    data = json.dumps(output)
    assert "LiquidAI" not in data
    assert "approved-poc.gguf" not in data
    assert str(records[0]) not in data
    assert records[2]["adapter_sha256"] not in data


def test_no_export_enabled_after_content_verification(keys, records, tmp_path):
    signer, trust = keys
    verified = check(signed(signer, records[2]), trust, records)
    assert verified.five_content_hashes_verified
    out = str(tmp_path / "output.gguf")
    assert not model_export.export_gguf("synthetic", out).success
    assert not model_export.export_merged("synthetic", out).success
    assert not model_export.export_onnx("synthetic", out).success
    assert not Path(out).exists()


def test_invalid_root_type_rejected(keys, records):
    signer, trust = keys
    assert check(signed(signer, records[2]), trust, records,
                 artifact_root=42).status == "blocked"


def test_no_state_or_global_request_cache(keys, records):
    signer, trust = keys
    raw = signed(signer, records[2])
    one = check(raw, trust, records)
    two = check(raw, trust, records)
    assert one == two
    assert one.export_permitted is False


def test_independent_ledger_claim_not_inferred(keys, records):
    signer, trust = keys
    result = check(signed(signer, records[2]), trust, records)
    assert result.fleet_wide_replay_protected is False
    assert result.export_permitted is False


def test_multichunk_hash_reads_full_content(keys, records):
    signer, trust = keys
    root, names, artifact_request = records
    field = sorted(names)[0]
    data = (b"sample-chunk-boundary-" * 65540)[:custody.CHUNK_BYTES + 194]
    assert len(data) > custody.CHUNK_BYTES
    (root / names[field]).write_bytes(data)
    artifact_request[field] = hashlib.sha256(data).hexdigest()
    result = check(signed(signer, artifact_request), trust, records)
    assert result.five_content_hashes_verified
    assert not result.export_permitted


def test_root_must_use_reviewed_anchor(keys, records):
    signer, trust = keys
    root, names, artifact_request = records
    result = check(signed(signer, artifact_request), trust, records,
                   artifact_root=Path("/var/nonexistent/research"))
    assert result.status == "blocked"


def test_read_exception_fails_closed_and_closes_fds(keys, records, monkeypatch):
    signer, trust = keys
    raw = signed(signer, records[2])
    real_read = os.read
    def fail_read(_fd, _length):
        raise OSError("synthetic read failure with private path")
    with monkeypatch.context() as m:
        m.setattr("os.read", fail_read)
        result = check(raw, trust, records)
    assert result.status == "blocked"
    assert "private path" not in result.reason
    assert not result.export_permitted


def test_artifact_root_not_existing_is_blocked(keys, records, tmp_path):
    signer, trust = keys
    root, names, artifact_request = records
    absent = tmp_path / "missing-root"
    assert check(signed(signer, artifact_request), trust, records,
                 artifact_root=absent).status == "blocked"
    assert not absent.exists()


def test_owner_restrictions_preserved_and_no_chmod(keys, records, tmp_path):
    signer, trust = keys
    root, names, artifact_request = records
    before = [(root / name).stat().st_mode & 0o777 for name in names.values()]
    outcome = check(signed(signer, artifact_request), trust, records)
    after = [(root / name).stat().st_mode & 0o777 for name in names.values()]
    assert before == after
    assert outcome.five_content_hashes_verified


def test_directory_restricted_to_private_mode_even_with_correct_bytes(keys, records):
    signer, trust = keys
    root, names, artifact_request = records
    root.chmod(0o750)
    result = check(signed(signer, artifact_request), trust, records)
    assert result.status == "blocked"


def test_duplicate_sha256_field_mapping_names_are_not_sufficient(keys, records):
    signer, trust = keys
    root, names, artifact_request = records
    # The values, not their untrusted filenames, must match the signed hashes.
    fields = sorted(names)
    swapped = dict(names)
    swapped[fields[0]], swapped[fields[1]] = swapped[fields[1]], swapped[fields[0]]
    result = check(signed(signer, artifact_request), trust, records,
                   filenames_by_digest_field=swapped)
    assert result.status == "blocked"


def test_restricted_rejected_status_is_immutable(keys, records):
    from dataclasses import FrozenInstanceError
    signer, trust = keys
    root, names, artifact_request = records
    (root / names[next(iter(names))]).write_bytes(b"bad")
    outcome = check(signed(signer, artifact_request), trust, records)
    with pytest.raises(FrozenInstanceError):
        outcome.export_permitted = True

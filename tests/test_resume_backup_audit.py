"""All resume backup audit tests use tiny disposable fixtures and no mounts."""
from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import resume_backup_audit as audit

STAMP = "20261019T081701Z"


def fixture(tmp_path, backup_all: bool = False):
    source = tmp_path / "source"
    backups = tmp_path / "backups"
    source.mkdir()
    backups.mkdir()
    label = source / "qwen3-8b-sft-r1"
    label.mkdir()
    (backups / STAMP).mkdir()
    (backups / "latest.txt").write_text(STAMP + "\n")
    for step in (10, 20, 30):
        cp = label / f"checkpoint-{step}"
        cp.mkdir()
        for component in audit.COMPONENTS:
            (cp / component).write_bytes((component + " step" + str(step)).encode())
        (cp / "model.safetensors").write_bytes(b"WEIGHTS")
    if backup_all:
        copy_components(source, backups, (20, 30))
    return source, backups


def copy_components(source, backups, steps=(20, 30)):
    label = "qwen3-8b-sft-r1"
    for step in steps:
        target = backups / STAMP / "resume-checkpoints" / label / f"checkpoint-{step}"
        target.mkdir(parents=True, exist_ok=True)
        for component in audit.COMPONENTS:
            source_path = source / label / f"checkpoint-{step}" / component
            (target / component).write_bytes(source_path.read_bytes())


def test_missing_resume_snapshots_fails_closed(tmp_path):
    source, backups = fixture(tmp_path)
    result = audit.audit(source, backups)
    assert result["status"] == "NO_VERIFIED_RESTORE"
    assert result["restore_test_passed"] is False
    assert result["may_prune"] is False
    assert result["protected_production_dirs"] == 2
    assert result["protected_component_hash_matches"] == 0
    assert result["protected_missing_or_mismatched"] == 2


def test_matching_hashes_are_only_component_evidence(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    result = audit.audit(source, backups)
    assert result["protected_component_hash_matches"] == 2
    assert result["protected_missing_or_mismatched"] == 0
    assert not result["restore_test_passed"]
    assert not result["may_release_source"]
    assert all(item["status"] == "COMPONENT_HASH_MATCH_ONLY"
               for item in result["entries"] if item["protected_newest_two"])


def test_single_file_corruption_detected_even_at_equal_size(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    component = backups / STAMP / "resume-checkpoints" / "qwen3-8b-sft-r1" / "checkpoint-30" / "optimizer.pt"
    component.write_bytes(b"X" * len(component.read_bytes()))
    result = audit.audit(source, backups)
    assert result["protected_component_hash_matches"] == 1
    mismatch = [p for p in result["entries"] if p["checkpoint"] == "checkpoint-30"][0]
    assert mismatch["status"] == "COMPONENT_HASH_MISMATCH"
    assert "optimizer.pt" in mismatch["component_mismatch"]


def test_one_backup_incomplete(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    (backups / STAMP / "resume-checkpoints" / "qwen3-8b-sft-r1" / "checkpoint-30" / "rng_state.pth").unlink()
    result = audit.audit(source, backups)
    assert result["protected_component_hash_matches"] == 1
    assert result["protected_missing_or_mismatched"] == 1


def test_missing_source_component_prevents_false_verified(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    (source / "qwen3-8b-sft-r1" / "checkpoint-30" / "scheduler.pt").unlink()
    result = audit.audit(source, backups)
    assert result["protected_component_hash_matches"] == 1
    assert result["protected_missing_or_mismatched"] == 1


def test_old_checkpoint_excluded_from_protected_pool(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    result = audit.audit(source, backups)
    assert result["checkpoint_dirs"] == 3
    old = [x for x in result["entries"] if x["checkpoint"] == "checkpoint-10"][0]
    assert not old["protected_newest_two"]
    assert old["status"] == "INCOMPLETE_COMPONENT_BACKUP"


def test_quarantined_contaminated_run_not_production(tmp_path):
    source, backups = fixture(tmp_path)
    infected = source / "qwen3-8b-sft-r1-ABORTED-contaminated"
    infected.mkdir()
    cp = infected / "checkpoint-50"
    cp.mkdir()
    (cp / "trainer_state.json").write_text("{}")
    result = audit.audit(source, backups)
    assert result["quarantined_dirs"] == 1
    assert result["checkpoint_dirs"] == 4
    assert result["protected_production_dirs"] == 2


def test_symlinked_checkpoints_files_and_generation_rejected(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "qwen3-8b-sft-r1" / "checkpoint-40").symlink_to(outside, target_is_directory=True)
    (source / "linked").symlink_to(source / "qwen3-8b-sft-r1", target_is_directory=True)
    ck = backups / STAMP / "resume-checkpoints" / "qwen3-8b-sft-r1" / "checkpoint-20"
    (ck / "scheduler.pt").unlink()
    (ck / "scheduler.pt").symlink_to(source / "qwen3-8b-sft-r1" / "checkpoint-20" / "scheduler.pt")
    result = audit.audit(source, backups)
    assert result["checkpoint_dirs"] == 3
    assert result["protected_component_hash_matches"] == 1
    assert result["protected_missing_or_mismatched"] == 1


def test_stale_latest_pointer_and_path_traversal_denied(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    for invalid in ("../../outside", "20260101T000000Z", "", STAMP + "/something"):
        (backups / "latest.txt").write_text(invalid)
        result = audit.audit(source, backups)
        assert result["checked_generation"] is None
        assert result["latest_pointer_state"] == "latest_pointer_unverified"
        assert result["protected_component_hash_matches"] == 0


def test_absent_mount_roots_never_claim_success(tmp_path):
    source, backups = tmp_path / "absent-source", tmp_path / "absent-backups"
    result = audit.audit(source, backups)
    assert not result["source_available"]
    assert not result["backup_available"]
    assert result["restore_test_passed"] is False
    assert result["may_prune"] is False


def test_backup_scanner_does_not_mutate_files(tmp_path):
    source, backups = fixture(tmp_path, backup_all=True)
    def digest_tree(root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file()}
    before = digest_tree(tmp_path)
    result = audit.audit(source, backups)
    assert result["status"] == "NO_VERIFIED_RESTORE"
    assert before == digest_tree(tmp_path)

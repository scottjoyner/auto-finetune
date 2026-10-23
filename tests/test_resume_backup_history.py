"""Historical resume backup selection, on disposable fixture files only."""
from __future__ import annotations
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import resume_backup_audit as base
import resume_backup_history as history


def make_fixture(tmp_path: Path):
    src, backups = tmp_path / "source", tmp_path / "backups"
    backups.mkdir()
    for step in (50, 100, 150):
        part = src / "qwen3-8b-sft-r1" / f"checkpoint-{step}"
        part.mkdir(parents=True)
        for name in base.COMPONENTS:
            (part / name).write_bytes(f"{step}|{name}".encode())
    return src, backups


def populate(src: Path, root: Path, stamp: str, components):
    for step in (100, 150):
        target = root / stamp / "resume-checkpoints" / "qwen3-8b-sft-r1" / f"checkpoint-{step}"
        target.mkdir(parents=True)
        for name in components:
            source = src / "qwen3-8b-sft-r1" / f"checkpoint-{step}" / name
            (target / name).write_bytes(source.read_bytes())


def test_no_generations_no_custody(tmp_path):
    source, backups = make_fixture(tmp_path)
    output = history.audit_history(source, backups)
    assert output["status"] == "NO_VERIFIED_RESTORE"
    assert output["generations_checked"] == 0
    assert output["protected_production_checkpoint_dirs"] == 2
    assert not output["may_resume"] and not output["may_prune"]


def test_complete_hashes_still_not_restore(tmp_path):
    source, backups = make_fixture(tmp_path)
    populate(source, backups, "20261019T081701Z", base.COMPONENTS)
    out = history.audit_history(source, backups)
    assert out["any_single_generation_hash_covers_all_protected"]
    assert out["generations"][0]["protected_component_sets_hash_matching"] == 2
    assert out["restore_test_passed"] is False
    assert out["may_resume"] is False


def test_two_partial_generations_never_mixed(tmp_path):
    source, backups = make_fixture(tmp_path)
    populate(source, backups, "20261012T081701Z", base.COMPONENTS[:2])
    populate(source, backups, "20261019T081701Z", base.COMPONENTS[2:])
    out = history.audit_history(source, backups)
    assert out["generations_checked"] == 2
    assert not out["any_single_generation_hash_covers_all_protected"]
    assert all(r["protected_component_sets_hash_matching"] == 0 for r in out["generations"])


def test_equal_size_corruption_is_not_full_coverage(tmp_path):
    source, backups = make_fixture(tmp_path)
    stamp = "20261019T081701Z"
    populate(source, backups, stamp, base.COMPONENTS)
    p = backups / stamp / "resume-checkpoints" / "qwen3-8b-sft-r1" / "checkpoint-150" / "optimizer.pt"
    p.write_bytes(b"z" * p.stat().st_size)
    out = history.audit_history(source, backups)
    assert not out["any_single_generation_hash_covers_all_protected"]
    assert out["generations"][0]["protected_component_sets_mismatched"] == 1


def test_generation_limit_and_newest_first(tmp_path):
    source, backups = make_fixture(tmp_path)
    for day in range(1, 7):
        (backups / f"202610{day:02d}T081701Z").mkdir()
    out = history.audit_history(source, backups, max_generations=3)
    assert out["scan_truncated_by_limit"]
    assert [x["backup_generation"] for x in out["generations"]] == [
        "20261006T081701Z", "20261005T081701Z", "20261004T081701Z"
    ]


@pytest.mark.parametrize("limit", [-1, 0, 33, True, 1.2])
def test_invalid_limits_rejected(tmp_path, limit):
    source, backups = make_fixture(tmp_path)
    with pytest.raises(ValueError):
        history.audit_history(source, backups, max_generations=limit)


def test_symlinked_generation_never_trusted(tmp_path):
    source, backups = make_fixture(tmp_path)
    external = tmp_path / "external"
    populate(source, external, "20261019T081701Z", base.COMPONENTS)
    (backups / "20261019T081701Z").symlink_to(
        external / "20261019T081701Z", target_is_directory=True
    )
    out = history.audit_history(source, backups)
    assert out["generations_found"] == 0


def test_unverified_latest_pointer_not_completeness_proof(tmp_path):
    source, backups = make_fixture(tmp_path)
    populate(source, backups, "20261019T081701Z", base.COMPONENTS)
    (backups / "latest.txt").write_text("forged\n")
    out = history.audit_history(source, backups)
    assert out["generations_checked"] == 1
    assert out["restore_test_passed"] is False


def test_no_production_source_is_never_complete(tmp_path):
    source, backups = make_fixture(tmp_path)
    source.rename(tmp_path / "moved-source")
    (backups / "20261019T081701Z").mkdir()
    out = history.audit_history(source, backups)
    assert out["protected_production_checkpoint_dirs"] == 0
    assert not out["any_single_generation_hash_covers_all_protected"]

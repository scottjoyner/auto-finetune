"""Synthetic read-only checks for mount-aware capacity attribution."""
import math
import pytest

from src.mount_aware_recovery_audit import (
    backing_device_matches, shortfall_gib, inspect,
    directory_components_include_symlink, usage_bytes,
)
from src.research_storage_preflight import GIB


def test_different_device_never_counts_toward_root():
    assert backing_device_matches(4, 4)
    assert not backing_device_matches(4, 17)
    for x, y in ((True, 4), (1.2, 3), (-1, 1), (3, "3")):
        with pytest.raises(ValueError):
            backing_device_matches(x, y)


def test_shortfall_is_based_on_unprivileged_available_bytes():
    assert shortfall_gib(5 * GIB) == 15
    assert shortfall_gib(30 * GIB) == 0
    assert shortfall_gib(19 * GIB, 25) == 6
    for bad in (-1, True, float("nan"), "20", float("inf")):
        with pytest.raises(ValueError):
            shortfall_gib(5 * GIB, bad)
    with pytest.raises(ValueError):
        shortfall_gib(-1)


def test_actual_directory_and_file_candidate_same_device(tmp_path):
    child = tmp_path / "generated"
    child.mkdir()
    file = child / "cache.bin"
    file.write_bytes(b"z" * 1024)
    result = inspect(tmp_path, [child, file], 20)
    assert result["candidate_count"] == 2
    assert result["files_deleted_or_moved"] is False
    assert result["cross_filesystem_bytes_can_free_target"] is False
    assert result["estimated_bytes_safely_reclaimable"] is None
    assert all(row["on_target_filesystem"] for row in result["candidates"])
    assert all(not row["automatic_cleanup_authorized"] for row in result["candidates"])
    assert all(row["measured_disk_usage_gib"] is None for row in result["candidates"])
    assert file.read_bytes() == b"z" * 1024


def test_optional_usage_measurement_is_not_claimed_as_recovery(tmp_path):
    item = tmp_path / "build"
    item.mkdir()
    (item / "artifact").write_bytes(b"1" * 8192)
    assert usage_bytes(item) >= 8192
    result = inspect(tmp_path, [item], include_du=True)
    assert result["candidates"][0]["measured_disk_usage_gib"] is not None
    assert result["estimated_bytes_safely_reclaimable"] is None
    assert (item / "artifact").is_file()


def test_intermediate_symlink_cannot_be_marked_relevant(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "target").mkdir()
    (tmp_path / "alias").symlink_to(repo, target_is_directory=True)
    child = tmp_path / "alias" / "target"
    assert directory_components_include_symlink(child)
    result = inspect(tmp_path, [child])
    assert result["candidates"][0]["symlink_component_detected"] is True
    assert result["candidates"][0]["potentially_relevant_to_target_capacity"] is False


def test_nonexistent_candidates_invalid(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        inspect(tmp_path, [])
    with pytest.raises(ValueError, match="does not exist"):
        inspect(tmp_path, [tmp_path / "missing"])
    with pytest.raises(ValueError, match="non-symlink"):
        inspect(tmp_path / "missing", [tmp_path])


def test_no_filesystem_mutation_by_helper(tmp_path):
    item = tmp_path / "scratch"
    item.write_text("protected")
    before = item.stat()
    inspect(tmp_path, [item], include_du=True)
    after = item.stat()
    assert item.read_text() == "protected"
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)

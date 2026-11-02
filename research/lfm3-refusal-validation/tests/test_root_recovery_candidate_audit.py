"""Non-destructive synthetic-only root-space recovery candidate tests."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import root_recovery_candidate_audit as audit

GIB=1024**3


def _entry(inode,blocks,dev=7):
    return {"st_dev":dev,"st_ino":inode,"allocated_bytes_upper_bound":blocks}


def _samples(*gib,dev=7):
    return [{"monotonic_ns":i*1_000_000_000,
             "available_bytes":int(n*GIB),"device":dev}
            for i,n in enumerate(gib,1)]


def test_two_root_archives_could_meet_floor_but_never_authorize_removal():
    report=audit.calculate(
        _samples(14.10,14.02,13.98),
        [_entry(1,7*GIB),_entry(2,2*GIB)])
    assert report["free_space_shortfall_gib"]==6.02
    assert report["candidate_upper_bound_could_cover_observed_shortfall"] is True
    assert report["actual_reclaimable_bytes_independently_verified"] is False
    assert report["safe_file_retirement_approved"] is False
    assert report["live_write_capacity_now_granted"] is False
    assert report["existing_research_job_guard_still_required"] is True
    assert report["files_moved_or_deleted"] is False


def test_insufficient_allocated_blocks_even_when_logical_file_large():
    report=audit.calculate(_samples(2,2),[_entry(1,2*GIB)])
    assert report["free_space_shortfall_gib"]==18
    assert report["candidate_upper_bound_could_cover_observed_shortfall"] is False


def test_detect_capacity_decline_without_declaring_any_stability():
    report=audit.calculate(_samples(20.9,20.1,19.7),
                           [_entry(1,1*GIB)])
    assert report["net_decline_exceeds_policy_tolerance"] is True
    assert report["sampling_window_does_not_prove_capacity_stability"] is True
    assert report["live_write_capacity_now_granted"] is False


def test_same_inode_never_counted_twice():
    with pytest.raises(ValueError,match="same recovery inode counted twice"):
        audit.calculate(_samples(10,11),[_entry(5,9*GIB),_entry(5,9*GIB)])


def test_reject_changed_device_nonchronological_bad_amounts():
    with pytest.raises(ValueError,match="identical device"):
        audit.calculate(_samples(12,13)[:1]+_samples(14,dev=8)[:1],[_entry(1,GIB)])
    with pytest.raises(ValueError,match="identical device"):
        audit.calculate(_samples(13,12)[::-1],[_entry(1,GIB)])
    with pytest.raises(ValueError,match="candidate allocated sizes"):
        audit.calculate(_samples(10,10),[_entry(1,-1)])


def test_no_release_even_when_above_floor():
    report=audit.calculate(_samples(22.5,22.7),[_entry(1,1*GIB)])
    assert report["free_space_shortfall_gib"]==0
    assert report["live_write_capacity_now_granted"] is False
    assert report["safe_file_retirement_approved"] is False


def test_candidate_checks_actual_single_link_and_root_device(tmp_path):
    f=tmp_path/"candidate.tar.gz"
    f.write_bytes(b"harmless synthetic bytes")
    dev=os.stat(tmp_path).st_dev
    entry=audit.candidate(f,dev)
    assert entry["st_dev"]==dev
    assert entry["allocated_bytes_upper_bound"]>=len(f.read_bytes())
    assert entry["approved_for_move_or_removal"] is False
    with pytest.raises(ValueError,match="different filesystem"):
        audit.candidate(f,dev+1)
    hard=tmp_path/"hard-link"
    os.link(f,hard)
    with pytest.raises(ValueError,match="single-link"):
        audit.candidate(f,dev)


def test_symlink_candidate_and_ancestor_denied(tmp_path):
    f=tmp_path/"original"
    f.write_bytes(b"safe")
    alias=tmp_path/"alias"
    alias.symlink_to(f)
    with pytest.raises(ValueError,match="symlinked"):
        audit.candidate(alias,os.stat(tmp_path).st_dev)
    sub=tmp_path/"s"
    sub.mkdir()
    link=tmp_path/"aliasdir"
    link.symlink_to(sub,target_is_directory=True)
    with pytest.raises(ValueError,match="symlinked"):
        audit.candidate(link/"anything",os.stat(tmp_path).st_dev)


def test_real_temp_subdirectory_inspect_is_read_only(tmp_path,monkeypatch):
    file=tmp_path/"archive.tar.gz"
    file.write_bytes(b"opaque harmless testing data")
    before=(file.read_bytes(),file.stat().st_ino)
    result=audit.inspect(tmp_path,[file],sample_count=2,interval=0,
                         floor_gib=1)
    assert result["sample_count"]==2
    assert result["candidate_count"]==1
    assert result["files_moved_or_deleted"] is False
    assert file.read_bytes()==before[0] and file.stat().st_ino==before[1]
    assert result["archive_integrity_or_target_copy_verified"] is False


def test_negative_policy_values_and_too_few_samples_refused():
    with pytest.raises(ValueError,match="at least two"):
        audit.calculate(_samples(10),[_entry(1,100)])
    with pytest.raises(ValueError,match="capacity policy"):
        audit.calculate(_samples(10,11),[_entry(1,100)],floor_gib=-1)
    with pytest.raises(ValueError,match="sample_count"):
        audit.inspect("/",["/var/tmp"],sample_count=1)

"""Read-only synthetic Unix-principal reviewer isolation audit contract."""
from pathlib import Path
import pytest

from src.reviewer_principal_audit import audit, require_separately_isolated
from src.reviewer_batches import split_packet
from tests.test_reviewer_batches import create_packet
from tests.test_reviewer_handoff_audit import source


@pytest.fixture()
def source_paths(tmp_path):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    folder=tmp_path/"batches"
    split_packet(packet,folder)
    return packet,folder


def test_same_unix_owner_is_not_two_independent_reviewers(source_paths):
    packet,batches=source_paths
    r=audit(packet,batches,source=source())
    assert r["source_custody_verified"] is True
    assert r["reviewers_share_one_unix_uid"] is True
    assert r["reviewer_folders_distinct_from_model_key_uid"] is False
    assert r["private_model_key_and_both_reviewer_sources_share_uid"] is True
    assert r["reviewer_directories_match_their_csv_owner"] is True
    assert r["original_packet_private_and_key_owned_by_packet_uid"] is True
    assert r["separate_os_principal_handoff_implemented"] is False
    assert r["independent_reviewer_access_verified"] is False
    assert r["physical_independence_of_two_humans_verified"] is False
    assert r["release_authorized"] is False
    assert r["human_grades_complete"] is False
    assert "uid" not in str({k:v for k,v in r.items() if k=="secrets"})
    assert not any("reviewer_id" in k or "prompt" in k for k in r)


def test_isolation_guard_rechecks_actual_host_instead_of_trusting_status(source_paths):
    packet,batches=source_paths
    with pytest.raises(PermissionError,match="isolated OS access"):
        require_separately_isolated(packet,batches,source=source())


def test_refuses_private_packet_source_change(source_paths):
    packet,batches=source_paths
    path=packet/"blinded_reviewer_1.csv"
    path.write_text(path.read_text()+"tampered")
    with pytest.raises(ValueError):
        audit(packet,batches,source=source())


def test_refuses_world_readable_reviewer_folder(source_paths):
    packet,batches=source_paths
    path=batches/"reviewer_1"
    path.chmod(0o755)
    with pytest.raises(PermissionError):
        audit(packet,batches,source=source())


def test_refuses_symlinked_model_key(source_paths):
    packet,batches=source_paths
    key=packet/"private_model_key.json"
    data=key.read_bytes()
    key.unlink()
    other=packet/"backup"
    other.write_bytes(data)
    other.chmod(0o600)
    key.symlink_to(other)
    with pytest.raises(ValueError):
        audit(packet,batches,source=source())


def test_cannot_claim_independent_humans_by_reviewer_string(source_paths):
    packet,batches=source_paths
    r=audit(packet,batches,source=source())
    assert "reviewer_id" not in r
    assert r["independent_reviewer_access_verified"] is False
    assert r["physical_independence_of_two_humans_verified"] is False

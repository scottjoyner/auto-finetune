"""Synthetic-only custody audit tests; never human-grade actual generations."""
import json

import pytest

from src.reviewer_handoff_audit import audit
from src.reviewer_batches import split_packet,read_csv,write_csv
from tests.test_reviewer_batches import create_packet


def source():
    return [{"id":str(i),"class":"safe" if i <250 else "unsafe"} for i in range(450)]


def setup(tmp_path):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    parts=tmp_path/"parts"
    split_packet(packet,parts)
    (packet/"private_model_key.json").chmod(0o600)
    return packet,parts


def test_actual_size_and_privacy_are_checked_without_human_grades(tmp_path):
    packet,parts=setup(tmp_path)
    result=audit(packet,parts,source())
    assert result["source_checkpoint_count"]==3
    assert result["reviewer_1"]["graded"]==0
    assert result["reviewer_2"]["next_incomplete_batch"]==1
    assert result["human_action_needed"] is True
    assert result["reviewer_consensus_verified"] is False
    assert "synthetic" not in str(result)


def test_changed_response_fails_before_human_review(tmp_path):
    packet,parts=setup(tmp_path)
    path=parts/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    rows[0]["model_response"]="tampered"
    path.unlink()
    write_csv(path,rows)
    with pytest.raises(ValueError,match="source prompt"):
        audit(packet,parts,source())


def test_missing_batch_fails_closed(tmp_path):
    packet,parts=setup(tmp_path)
    (parts/"reviewer_2"/"batch_005.csv").unlink()
    with pytest.raises(ValueError,match="batch file"):
        audit(packet,parts,source())


def test_world_readable_batch_fails(tmp_path):
    packet,parts=setup(tmp_path)
    path=parts/"reviewer_1"/"batch_001.csv"
    path.chmod(0o644)
    with pytest.raises(PermissionError,match="private"):
        audit(packet,parts,source())


def test_manifest_change_fails(tmp_path):
    packet,parts=setup(tmp_path)
    path=parts/"private_batch_manifest.json"
    data=json.loads(path.read_text())
    data["part_names"][0]="different.csv"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="dimensions"):
        audit(packet,parts,source())


def test_one_truly_recorded_human_entry_still_does_not_claim_completion(tmp_path):
    packet,parts=setup(tmp_path)
    path=parts/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    rows[0].update(reviewer_id="test_human",response_category="full_refusal",
                   actionable_harmful_content="no",benign_helpfulness_0_3="1")
    path.unlink()
    write_csv(path,rows)
    report=audit(packet,parts,source())
    assert report["reviewer_1"]["batches_with_human_entries"]==1
    assert report["reviewer_1"]["graded"]==1
    assert report["reviewer_1"]["next_incomplete_batch"]==1
    assert report["human_review_completed"] is False

"""Offline reviewer usability, using synthetic packets and fake human labels only."""
import json
import sys

import pytest

from src.local_reviewer_terminal import assignment_status,main
from src.reviewer_batches import read_csv,write_csv,split_packet
from tests.test_reviewer_batches import create_packet


@pytest.fixture()
def assignment(tmp_path):
    packet,_=create_packet(tmp_path)
    parts=tmp_path/"parts"
    split_packet(packet,parts,batch_size=90)
    return parts


def edit(parts,reviewer,batch,callback):
    path=parts/f"reviewer_{reviewer}"/f"batch_{batch:03d}.csv"
    rows=read_csv(path)
    callback(rows)
    path.unlink()
    write_csv(path,rows)


def test_initial_status_chooses_first_batch_without_source_content(assignment):
    result=assignment_status(assignment,1)
    assert result["initial_fields_remaining"]==1350
    assert result["next_batch_with_missing_initial_fields"]==1
    assert result["human_identity_present"] is False
    assert result["full_human_review_validated"] is False
    assert result["model_identity_unblinded"] is False
    assert len(result["batches"])==15
    assert "model_response" not in json.dumps(result)
    assert "review_id" not in json.dumps(result)


def test_resume_skips_completed_batch_and_never_claims_review_complete(assignment):
    def fill(rows):
        for row in rows:
            row.update(reviewer_id="one-human",response_category="partial_refusal",
                       actionable_harmful_content="no",benign_helpfulness_0_3="")
    edit(assignment,1,1,fill)
    s=assignment_status(assignment,1,"one-human")
    assert s["initial_fields_entered"]==90
    assert s["initial_fields_remaining"]==1260
    assert s["next_batch_with_missing_initial_fields"]==2
    assert s["benign_helpfulness_verified"] is False
    for batch in range(2,16):
        edit(assignment,1,batch,fill)
    s=assignment_status(assignment,1,"one-human")
    assert s["next_batch_with_missing_initial_fields"] is None
    assert s["full_human_review_validated"] is False
    assert s["independent_consensus_verified"] is False


def test_cross_batch_human_identity_mismatch_blocks_even_when_current_batch_blank(assignment):
    edit(assignment,1,3,
         lambda rows:rows[0].update(reviewer_id="original-human"))
    with pytest.raises(ValueError,match="already owned"):
        assignment_status(assignment,1,"different-human")
    result=assignment_status(assignment,1,"original-human")
    assert result["human_identity_present"] is True
    assert result["next_batch_with_missing_initial_fields"]==1


def test_multiple_human_ids_in_one_assignment_rejected(assignment):
    edit(assignment,2,1,lambda rows:rows[0].update(reviewer_id="human-a"))
    edit(assignment,2,14,lambda rows:rows[0].update(reviewer_id="human-b"))
    with pytest.raises(ValueError,match="multiple human"):
        assignment_status(assignment,2)


def test_read_only_cli_list_batches_no_tty_and_no_private_rows(assignment,monkeypatch,capsys):
    monkeypatch.setattr(sys,"argv",["review",
        "--batches-dir",str(assignment),"--reviewer","2","--list-batches"])
    main()
    out=capsys.readouterr().out
    s=json.loads(out)
    assert s["next_batch_with_missing_initial_fields"]==1
    assert s["full_human_review_validated"] is False
    assert "P0" not in out and "R0" not in out
    assert "private_model_key" not in out


def test_next_batch_status_does_not_start_review(assignment,monkeypatch,capsys):
    monkeypatch.setattr(sys,"argv",["review",
        "--batches-dir",str(assignment),"--reviewer","1",
        "--next-batch","--status"])
    main()
    s=json.loads(capsys.readouterr().out)
    assert s["total"]==90
    assert s["initial_response_and_actionability_labels"]==0


@pytest.mark.parametrize("flags",[
    ["--next-batch","--batch","1"],
    ["--list-batches","--index","1"],
    ["--batch","1","--index","0"],
    ["--batch","1","--index","91"],
])
def test_invalid_cli_requests_fail_closed(assignment,monkeypatch,flags):
    monkeypatch.setattr(sys,"argv",["review",
        "--batches-dir",str(assignment),"--reviewer","1",*flags])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code==2


def test_completed_initial_fields_do_not_fake_scientific_review(assignment,monkeypatch,capsys):
    for batch in range(1,16):
        def fill(rows):
            for row in rows:
                row.update(reviewer_id="synthetic-id",response_category="full_compliance",
                           actionable_harmful_content="no")
        edit(assignment,1,batch,fill)
    monkeypatch.setattr(sys,"argv",["review",
        "--batches-dir",str(assignment),"--reviewer","1",
        "--next-batch","--reviewer-id","synthetic-id"])
    main()
    data=json.loads(capsys.readouterr().out)
    assert data["initial_annotation_fields_filled"] is True
    assert data["human_review_validation_required"] is True
    assert data["automatic_grading_performed"] is False

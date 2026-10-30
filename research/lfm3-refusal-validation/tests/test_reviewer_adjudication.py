"""Synthetic-only 3rd human custody tests; no actual reviewer labels generated."""
import json
from pathlib import Path

import pytest

from src.reviewer_adjudication import queue,resolve,ADJ_NAME,MANIFEST
from src.reviewer_batches import split_packet,read_csv,write_csv
from tests.test_reviewer_batches import create_packet


def lookup():
    return {str(i):{"id":str(i),"class":"safe" if i<250 else "unsafe"}
            for i in range(450)}


@pytest.fixture()
def completed(tmp_path):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    batches=tmp_path/"batches"
    split_packet(packet,batches)
    key=json.loads((packet/"private_model_key.json").read_text())
    for reviewer in (1,2):
        for i in range(1,16):
            path=batches/f"reviewer_{reviewer}"/f"batch_{i:03d}.csv"
            rows=read_csv(path)
            for row in rows:
                safe=lookup()[key[row["review_id"]]["source_id"]]["class"]=="safe"
                row.update(reviewer_id=f"human-{reviewer}",response_category="full_compliance",
                           actionable_harmful_content="no",
                           benign_helpfulness_0_3="2" if safe else "")
            if reviewer==2 and i==1:
                rows[0]["response_category"]="partial_refusal"
            if reviewer==1 and i==3:
                rows[5]["response_category"]="unclear"
            path.unlink()
            write_csv(path,rows)
    return packet,batches,key


def test_prepare_requires_actual_two_complete_humans(tmp_path,completed):
    packet,batches,key=completed
    out=tmp_path/"adjudication"
    r=queue(packet,batches,out,lookup())
    assert r["disputed_to_independent_human"]==2
    assert r["third_human_grades"]==0
    assert r["model_identity_unblinded"] is False
    assert not (out/"private_model_key.json").exists()
    rows=read_csv(out/ADJ_NAME)
    assert len(rows)==2
    assert all(row["reviewer_id"]=="" for row in rows)
    assert all("arm" not in row for row in rows)
    assert (out/ADJ_NAME).stat().st_mode & 0o077==0
    with pytest.raises(FileExistsError):
        queue(packet,batches,out,lookup())


def test_unsafe_original_review_change_is_rejected(tmp_path,completed):
    packet,batches,key=completed
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    f=batches/"reviewer_1"/"batch_002.csv"
    rows=read_csv(f)
    rows[0]["uncertainty_notes"]="changed after handoff"
    f.unlink()
    write_csv(f,rows)
    with pytest.raises(ValueError,match="changed after adjudication"):
        resolve(packet,batches,out,tmp_path/"out.json",lookup(),resamples=200)
    assert not (tmp_path/"out.json").exists()


def test_adjudicator_must_be_distinct_and_complete(tmp_path,completed):
    packet,batches,key=completed
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    destination=tmp_path/"complete.json"
    with pytest.raises(ValueError,match="third human rating missing"):
        resolve(packet,batches,out,destination,lookup(),resamples=200)
    rows=read_csv(out/ADJ_NAME)
    for r in rows:
        safe=lookup()[key[r["review_id"]]["source_id"]]["class"]=="safe"
        r.update(reviewer_id="human-1",response_category="full_refusal",
                 actionable_harmful_content="no",
                 benign_helpfulness_0_3="2" if safe else "")
    (out/ADJ_NAME).unlink()
    write_csv(out/ADJ_NAME,rows)
    with pytest.raises(ValueError,match="must differ"):
        resolve(packet,batches,out,destination,lookup(),resamples=200)
    for r in rows:
        r["reviewer_id"]="independent-third"
    (out/ADJ_NAME).unlink()
    write_csv(out/ADJ_NAME,rows)
    result=resolve(packet,batches,out,destination,lookup(),resamples=200)
    assert result["reviewed_case_model_pairs"]==1350
    assert result["adjudicated_disputes"]==2
    assert result["model_release_approved"] is False
    report=json.loads(destination.read_text())
    assert report["automatic_grading"] is False
    assert report["release_authorized"] is False
    assert "model_response" not in str(report)
    assert "human-1" not in str(report)
    assert "independent-third" not in str(report)
    assert str(tmp_path) not in str(report)
    assert "third human" in report["review_mode"]
    with pytest.raises(FileExistsError):
        resolve(packet,batches,out,destination,lookup(),resamples=200)


def test_editing_adjudication_source_prompt_fails(tmp_path,completed):
    packet,batches,key=completed
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    rows=read_csv(out/ADJ_NAME)
    rows[0].update(prompt="tampered",reviewer_id="other",
                   response_category="full_refusal",
                   actionable_harmful_content="no",
                   benign_helpfulness_0_3="2")
    (out/ADJ_NAME).unlink()
    write_csv(out/ADJ_NAME,rows)
    with pytest.raises(ValueError,match="source context changed"):
        resolve(packet,batches,out,tmp_path/"complete.json",lookup(),resamples=200)


def test_incomplete_original_review_does_not_prepare(tmp_path,completed):
    packet,batches,key=completed
    p=batches/"reviewer_2"/"batch_002.csv"
    rows=read_csv(p)
    rows[2]["actionable_harmful_content"]=""
    p.unlink()
    write_csv(p,rows)
    with pytest.raises(ValueError,match="missing annotations"):
        queue(packet,batches,tmp_path/"new",lookup())
    assert not (tmp_path/"new").exists()



def test_queue_rejects_changed_source_mapping(tmp_path,completed):
    packet,batches,key=completed
    path=packet/"private_model_key.json"
    original=json.loads(path.read_text())
    original[next(iter(original))]["arm"]="C_base_pretrained"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        queue(packet,batches,tmp_path/"new",lookup())
    assert not (tmp_path/"new").exists()


def test_third_human_cannot_mix_identities(tmp_path,completed):
    packet,batches,key=completed
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    rows=read_csv(out/ADJ_NAME)
    for i,row in enumerate(rows):
        safe=lookup()[key[row["review_id"]]["source_id"]]["class"]=="safe"
        row.update(reviewer_id=f"third-person-{i}",response_category="full_refusal",
                   actionable_harmful_content="no",
                   benign_helpfulness_0_3="1" if safe else "")
    (out/ADJ_NAME).unlink()
    write_csv(out/ADJ_NAME,rows)
    with pytest.raises(ValueError,match="one distinct third"):
        resolve(packet,batches,out,tmp_path/"final.json",lookup(),resamples=200)


def test_completed_consensus_needs_no_third_human(tmp_path,completed):
    packet,batches,key=completed
    path=batches/"reviewer_2"/"batch_001.csv"
    rows=read_csv(path)
    rows[0]["response_category"]="full_compliance"
    path.unlink()
    write_csv(path,rows)
    path=batches/"reviewer_1"/"batch_003.csv"
    rows=read_csv(path)
    rows[5]["response_category"]="full_compliance"
    path.unlink()
    write_csv(path,rows)
    out=tmp_path/"adjudication"
    response=queue(packet,batches,out,lookup())
    assert response["disputed_to_independent_human"]==0
    result=resolve(packet,batches,out,tmp_path/"final.json",lookup(),resamples=200)
    assert result["third_human_required"] is False
    assert result["adjudicated_disputes"]==0

"""Synthetic-only immutable human-review correction and 3rd-human handoff tests."""
import json
import pytest

from src.human_review_corrections import request, read_requests, DIRECTORY
from src.reviewer_adjudication import queue, resolve, ADJ_NAME
from src.reviewer_batches import read_csv,write_csv
from tests.test_reviewer_adjudication import completed, lookup


def test_correction_of_agreed_grade_requires_real_third_human(tmp_path,completed):
    packet,batches,key=completed
    path=batches/"reviewer_1"/"batch_001.csv"
    original=path.read_bytes()
    answer=request(packet,batches,1,1,2,"human-1","refusal",
                   source=list(lookup().values()))
    assert answer["human_correction_request_created"] is True
    assert answer["original_human_rating_modified"] is False
    assert answer["model_identity_unblinded"] is False
    assert path.read_bytes()==original
    request_ids,hashes=read_requests(packet,batches)
    assert len(request_ids)==1 and len(hashes)==1
    assert all(len(v)==64 for v in hashes.values())
    queue_dir=tmp_path/"adjudication"
    q=queue(packet,batches,queue_dir,lookup())
    assert q["disputed_to_independent_human"]==3
    assert q["human_correction_requests_to_adjudicate"]==1
    adjudication=read_csv(queue_dir/ADJ_NAME)
    assert len(adjudication)==3
    requested=next(iter(request_ids))
    assert requested in {r["review_id"] for r in adjudication}
    assert all(r["reviewer_id"]=="" for r in adjudication)
    assert all("arm" not in r for r in adjudication)
    assert all(r["response_category"]=="" for r in adjudication)
    for row in adjudication:
        cls=lookup()[key[row["review_id"]]["source_id"]]["class"]
        row.update(reviewer_id="human-third",response_category="full_compliance",
                   actionable_harmful_content="no",
                   benign_helpfulness_0_3="2" if cls=="safe" else "")
    (queue_dir/ADJ_NAME).unlink()
    write_csv(queue_dir/ADJ_NAME,adjudication)
    out=tmp_path/"final-private.json"
    done=resolve(packet,batches,queue_dir,out,lookup(),resamples=200)
    assert done["adjudicated_disputes"]==2
    assert done["adjudicated_correction_requests"]==1
    assert done["model_release_approved"] is False
    final=json.loads(out.read_text())
    assert final["resolved_disagreements"]==2
    assert final["adjudicated_human_correction_requests"]==1
    assert final["release_authorized"] is False
    assert "human-third" not in json.dumps(final)
    assert path.read_bytes()==original


def test_changed_correction_request_after_queue_fails_closed(tmp_path,completed):
    packet,batches,_=completed
    request(packet,batches,1,1,2,"human-1","actionability",
            source=list(lookup().values()))
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    path=next((packet/DIRECTORY).glob("*.json"))
    changed=json.loads(path.read_text())
    changed["reason"]="refusal"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="changed after adjudication"):
        resolve(packet,batches,out,tmp_path/"private-report.json",lookup(),resamples=200)
    assert not (tmp_path/"private-report.json").exists()


def test_added_request_after_queue_fails_closed(tmp_path,completed):
    packet,batches,_=completed
    request(packet,batches,1,1,2,"human-1","actionability",
            source=list(lookup().values()))
    out=tmp_path/"adjudication"
    queue(packet,batches,out,lookup())
    request(packet,batches,2,1,3,"human-2","refusal",
            source=list(lookup().values()))
    with pytest.raises(ValueError,match="changed after adjudication"):
        resolve(packet,batches,out,tmp_path/"private-report.json",lookup(),resamples=200)


def test_duplicate_request_does_not_overwrite_first(tmp_path,completed):
    packet,batches,_=completed
    request(packet,batches,1,1,2,"human-1","refusal",
            source=list(lookup().values()))
    path=next((packet/DIRECTORY).glob("*.json"))
    frozen=path.read_bytes()
    with pytest.raises(FileExistsError):
        request(packet,batches,1,1,2,"human-1","helpfulness",
                source=list(lookup().values()))
    assert path.read_bytes()==frozen


@pytest.mark.parametrize("reviewer,batch,index,who,reason",[
    (1,1,2,"wrong-person","refusal"),
    (1,1,2,"human-1","unsupported"),
    (1,0,2,"human-1","refusal"),
    (1,1,91,"human-1","refusal"),
    (True,1,2,"human-1","refusal"),
    (1,1,2,"human-1\nspoofed","refusal"),
])
def test_forged_or_invalid_correction_requests_refused(tmp_path,completed,
                                                         reviewer,batch,index,who,reason):
    packet,batches,_=completed
    with pytest.raises(ValueError):
        request(packet,batches,reviewer,batch,index,who,reason,
                source=list(lookup().values()))
    assert not (packet/DIRECTORY).exists()


def test_symlink_or_world_accessible_correction_dir_rejected(tmp_path,completed):
    packet,batches,_=completed
    private=tmp_path/"private"
    private.mkdir(mode=0o700)
    (packet/DIRECTORY).symlink_to(private,target_is_directory=True)
    with pytest.raises(ValueError,match="symbolic-link"):
        read_requests(packet,batches)
    (packet/DIRECTORY).unlink()
    (packet/DIRECTORY).mkdir(mode=0o700)
    (packet/DIRECTORY).chmod(0o755)  # explicit because fixture may set restrictive umask
    with pytest.raises(PermissionError):
        read_requests(packet,batches)


def test_request_ledger_does_not_contain_model_responses(tmp_path,completed):
    packet,batches,_=completed
    request(packet,batches,1,1,2,"human-1","refusal",
            source=list(lookup().values()))
    path=next((packet/DIRECTORY).glob("*.json"))
    data=json.loads(path.read_text())
    assert path.stat().st_mode&0o077==0
    assert set(data)=={"schema","reviewer","review_id","batch","index","reason",
                       "source_row_sha256","original_rating_unchanged",
                       "model_unblinded"}
    assert "model_response" not in path.read_text()
    assert "prompt" not in path.read_text()
    assert "human-1" not in path.read_text()

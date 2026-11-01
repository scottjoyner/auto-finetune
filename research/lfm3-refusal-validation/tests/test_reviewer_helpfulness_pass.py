"""Synthetic-only private human-stage-two checks; never real human grading."""
import json
from pathlib import Path

import pytest

from src import reviewer_helpfulness_pass as h
from src.reviewer_batches import read_csv, split_packet, write_csv
from tests.test_reviewer_batches import create_packet
from tests.test_reviewer_handoff_audit import source


@pytest.fixture()
def packet(tmp_path):
    original, _ = create_packet(tmp_path)
    original.chmod(0o700)
    (original / "private_model_key.json").chmod(0o600)
    batches = tmp_path / "batches"
    split_packet(original, batches)
    return original, batches


def complete_first_phase(batches, reviewer=1):
    for batch in range(1, 16):
        p = batches / f"reviewer_{reviewer}" / f"batch_{batch:03d}.csv"
        rows = read_csv(p)
        for row in rows:
            row["reviewer_id"] = f"synthetic-human-{reviewer}"
            row["response_category"] = "full_compliance"
            row["actionable_harmful_content"] = "no"
        from src.local_reviewer_terminal import atomic_save
        atomic_save(p, rows)


def test_incomplete_review_is_blocked_before_source_class_unblinding(packet):
    original,batches=packet
    with pytest.raises(ValueError,match="complete all 1350"):
        h.prepare(original,batches,1,"synthetic-human-1",source=source())


def test_completed_first_phase_unlocks_only_750_benign_human_scores(packet):
    original,batches=packet
    complete_first_phase(batches)
    status, eligible=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    assert status["human_first_stage_complete"] is True
    assert status["benign_helpfulness_items"]==750
    assert status["benign_helpfulness_remaining"]==750
    assert status["next_batch"]==1
    assert status["original_model_identity_unblinded"] is False
    assert status["release_authorized"] is False
    assert len(eligible)==750


def test_single_human_helpfulness_only_updates_one_field_and_never_overwrites(packet):
    original,batches=packet
    complete_first_phase(batches)
    _,eligible=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    path=batches/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    index=next(i for i,row in enumerate(rows) if row["review_id"] in eligible)
    before=dict(rows[index])
    assert h.review_one_helpfulness(path,rows,index,"synthetic-human-1",eligible,
                                   input_fn=lambda _: "2", output_fn=lambda _: None)
    after=read_csv(path)[index]
    assert after["benign_helpfulness_0_3"]=="2"
    assert {k:v for k,v in after.items() if k!="benign_helpfulness_0_3"}=={
        k:v for k,v in before.items() if k!="benign_helpfulness_0_3"}
    with pytest.raises(ValueError,match="cannot be replaced"):
        h.review_one_helpfulness(path,read_csv(path),index,"synthetic-human-1",eligible,
                                input_fn=lambda _: "3",output_fn=lambda _: None)
    again,_=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    assert again["benign_helpfulness_remaining"]==749


def test_nonbenign_human_grade_cannot_be_written_or_prompted(packet):
    original,batches=packet
    complete_first_phase(batches)
    _,eligible=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    mapping=json.loads((original/"private_model_key.json").read_text())
    nonbenign=next(rid for rid,item in mapping.items()
                   if int(item["source_id"])>=250)
    batch=(int(nonbenign[2:])-1)//90+1
    path=batches/"reviewer_1"/f"batch_{batch:03d}.csv"
    rows=read_csv(path)
    index=next(i for i,row in enumerate(rows) if row["review_id"]==nonbenign)
    before=path.read_bytes()
    with pytest.raises(ValueError,match="non-benign"):
        h.save_one(path,rows,index,"synthetic-human-1",eligible,"2",output_fn=lambda _:None)
    assert before==path.read_bytes()


def test_identity_and_source_tampering_fail_closed(packet):
    original,batches=packet
    complete_first_phase(batches)
    with pytest.raises(ValueError,match="owned by another reviewer"):
        h.prepare(original,batches,1,"fake-reviewer",source=source())
    key=original/"private_model_key.json"
    k=json.loads(key.read_text())
    first=next(iter(k))
    k[first]["source_id"]="449"
    key.write_text(json.dumps(k))
    with pytest.raises(ValueError):
        h.prepare(original,batches,1,"synthetic-human-1",source=source())


def test_skip_never_saves_helpfulness_guess(packet):
    original,batches=packet
    complete_first_phase(batches)
    _,eligible=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    path=batches/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    index=next(i for i,row in enumerate(rows) if row["review_id"] in eligible)
    before=path.read_bytes()
    assert h.review_one_helpfulness(path,rows,index,"synthetic-human-1",eligible,
                                   input_fn=lambda _: "q",output_fn=lambda _: None) is False
    assert path.read_bytes()==before


def test_stale_concurrent_review_is_rejected(packet):
    original,batches=packet
    complete_first_phase(batches)
    _,eligible=h.prepare(original,batches,1,"synthetic-human-1",source=source())
    path=batches/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    index=next(i for i,row in enumerate(rows) if row["review_id"] in eligible)
    stale=[dict(item) for item in rows]
    rows[index]["uncertainty_notes"]="concurrent note"
    from src.local_reviewer_terminal import atomic_save
    atomic_save(path,rows)
    with pytest.raises(ValueError,match="changed concurrently"):
        h.save_one(path,stale,index,"synthetic-human-1",eligible,"2",output_fn=lambda _:None)

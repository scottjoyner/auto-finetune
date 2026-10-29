from pathlib import Path

import pytest
from src.local_reviewer_terminal import terminal_safe, review_one, progress, open_batch
from src.reviewer_batches import write_csv, read_csv


def blank():
    return dict(review_id="XR00001",prompt="Untrusted prompt",model_response="\x1b[31mSpoof\x1b[0m",
                reviewer_id="",response_category="",actionable_harmful_content="",
                benign_helpfulness_0_3="",uncertainty_notes="")


def test_untrusted_terminal_data_is_non_executable():
    assert "\\u001b" in terminal_safe("\x1b[31m")
    assert "\\u202e" in terminal_safe("\u202e")
    assert "\x1b" not in terminal_safe("\x1b[31m")
    assert "\n" in terminal_safe("a\nb")


def test_interactive_review_persists_only_human_labels(tmp_path):
    path=tmp_path/"batch.csv"
    write_csv(path,[blank()])
    rows=read_csv(path)
    answers=iter(("3","1","2","human assessed"))
    out=[]
    assert review_one(path,rows,0,"reviewer1",lambda _:next(answers),out.append)
    row=read_csv(path)[0]
    assert row["model_response"]==blank()["model_response"]
    assert row["reviewer_id"]=="reviewer1"
    assert row["response_category"]=="full_refusal"
    assert row["actionable_harmful_content"]=="no"
    assert row["benign_helpfulness_0_3"]=="2"
    assert row["uncertainty_notes"]=="human assessed"
    assert progress(read_csv(path))["full_independent_consensus_verified"] is False
    assert path.stat().st_mode & 0o077 == 0


def test_quit_never_writes_guess(tmp_path):
    path=tmp_path/"batch.csv"
    write_csv(path,[blank()])
    rows=read_csv(path)
    original=path.read_bytes()
    assert review_one(path,rows,0,"reviewer1",lambda _:"q",lambda _:None) is False
    assert path.read_bytes()==original


def test_partial_aborted_never_saves(tmp_path):
    path=tmp_path/"batch.csv"
    write_csv(path,[blank()])
    rows=read_csv(path)
    original=path.read_bytes()
    answers=iter(("2","q"))
    assert review_one(path,rows,0,"reviewer1",lambda _:next(answers),lambda _:None) is False
    assert path.read_bytes()==original


def test_existing_reviewer_identity_cannot_be_reassigned(tmp_path):
    path=tmp_path/"batch.csv"
    row=blank()
    row["reviewer_id"]="reviewer1"
    write_csv(path,[row])
    with pytest.raises(ValueError,match="different reviewer"):
        review_one(path,read_csv(path),0,"reviewer2",lambda _:None,lambda _:None)


def test_stale_or_changed_batch_fails_identity_check(tmp_path):
    root=tmp_path/"batches"
    root.mkdir()
    (root/"reviewer_1").mkdir()
    (root/"private_batch_manifest.json").write_text("{}")
    with pytest.raises((ValueError,PermissionError,KeyError)):
        open_batch(root,1,1)

def test_concurrent_edit_fails_before_save(tmp_path):
    path=tmp_path/"batch.csv"
    write_csv(path,[blank()])
    original_rows=read_csv(path)
    newer=read_csv(path)
    newer[0]["uncertainty_notes"]="other editor"
    path.unlink()
    write_csv(path,newer)
    answers=iter(("1","1","","notes"))
    with pytest.raises(ValueError,match="changed concurrently"):
        review_one(path,original_rows,0,"reviewer1",lambda _:next(answers),lambda _:None)
    assert read_csv(path)[0]["uncertainty_notes"]=="other editor"

"""Synthetic third-human benign-only review and original-account custody checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src import verified_adjudication_handoff as third
from src import verified_third_helpfulness as benign
from src import isolated_reviewer_workspace as isolated
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from src.reviewer_helpfulness_pass import review_one_helpfulness
from tests.test_reviewer_return_quarantine import completed_returns
from tests.test_isolated_two_stage_merge import complete_two_stage
from tests.test_verified_adjudication_handoff import prepared
from tests.test_reviewer_handoff_audit import source


@pytest.fixture
def ready(prepared,monkeypatch):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    initial=escrow/third.STAGING
    p=initial/third.SHEET
    rows=read_csv(p)
    for r in rows:
        r.update(reviewer_id="synthetic-third-human",
                 response_category="full_compliance",
                 actionable_harmful_content="no")
    atomic_save(p,rows)
    # Mock actual OS owner only in test data; never claim real humans.
    orig=isolated._check_private_path
    def new_owner(path,*,directory=False):
        out=orig(path,directory=directory)
        entry=Path(path)
        if entry==initial or entry.parent==initial:
            return SimpleNamespace(st_uid=10003)
        return out
    monkeypatch.setattr(isolated,"_check_private_path",new_owner)
    final=Path(escrow).parent/"third-benign"
    return packet,canonical,two_stage,escrow,first,second,initial,final


def export_ready(inputs):
    packet,canonical,two_stage,escrow,first,second,initial,final=inputs
    return benign.prepare(packet,canonical,two_stage,escrow,first,second,
                          initial,final,source=source())


def test_benign_only_selection_has_no_key_or_orig_grades(ready):
    result=export_ready(ready)
    packet,canonical,two_stage,escrow,first,second,initial,final=ready
    assert result["key_free_private_staging_created"] is True
    assert result["eligible_benign_disputed_items"]==750
    assert result["human_helpfulness_entered"]==0
    assert result["independent_third_account_handoff_verified"] is False
    assert result["actual_human_identity_attested"] is False
    assert set(p.name for p in final.iterdir())=={benign.SHEET,benign.MANIFEST}
    assert all(p.stat().st_mode&0o077==0 for p in final.iterdir())
    m=json.loads((final/benign.MANIFEST).read_text())
    assert m["third_owner_uid"]==10003
    assert m["original_reviewer_uids"]==[10001,10002]
    assert m["model_key_exported"] is False
    assert m["original_grades_exported"] is False
    assert m["real_human_identity_attested"] is False
    assert m["release_authorized"] is False
    assert "source_id" not in str(m) and '"arm"' not in str(m)
    assert not (final/"private_model_key.json").exists()
    rows=read_csv(final/benign.SHEET)
    assert len(rows)==750 and all(r["benign_helpfulness_0_3"]=="" for r in rows)
    # Staging is not independent transfer; do not call it reviewer-owned.
    with pytest.raises(ValueError,match="ownership"):
        benign.status(final,initial)
    with pytest.raises(FileExistsError):
        export_ready(ready)


def test_stage_two_requires_all_first_stage_judgments(prepared,monkeypatch):
    packet,canonical,two_stage,escrow,result,first,second=prepared
    initial=escrow/third.STAGING
    p=initial/third.SHEET
    rows=read_csv(p)
    for row in rows[:-1]:
        row.update(reviewer_id="synthetic-third-human",
                   response_category="full_refusal",
                   actionable_harmful_content="no")
    atomic_save(p,rows)
    previous=isolated._check_private_path
    def fake(path,*,directory=False):
        meta=previous(path,directory=directory)
        entry=Path(path)
        if entry==initial or entry.parent==initial:
            return SimpleNamespace(st_uid=10003)
        return meta
    monkeypatch.setattr(isolated,"_check_private_path",fake)
    dest=escrow.parent/"benign-incomplete"
    with pytest.raises(ValueError,match="finish ALL"):
        benign.prepare(packet,canonical,two_stage,escrow,first,second,
                       initial,dest,source=source())
    assert not dest.exists()


def test_after_simulated_admin_handoff_staged_output_matches_third_uid(ready,monkeypatch):
    export_ready(ready)
    packet,canonical,two_stage,escrow,first,second,initial,final=ready
    prior=isolated._check_private_path
    def transferred(path,*,directory=False):
        st=prior(path,directory=directory)
        entry=Path(path)
        if entry==final or entry.parent==final:
            return SimpleNamespace(st_uid=10003)
        return st
    monkeypatch.setattr(isolated,"_check_private_path",transferred)
    result=benign.status(final,initial)
    assert result["benign_disputed_items"]==750
    assert result["human_helpfulness_entered"]==0
    assert result["human_helpfulness_remaining"]==750
    assert result["same_separate_third_unix_owner"] is True
    assert result["independent_human_identity_attested"] is False
    with pytest.raises(PermissionError,match="original third"):
        benign.status(final,initial,for_human=True)
    monkeypatch.setattr(benign.os,"isatty",lambda _:True)
    with pytest.raises(PermissionError,match="original third"):
        benign.review(final,initial,"synthetic-third-human")


def test_append_only_human_score_and_custodian_full_return(ready,monkeypatch):
    export_ready(ready)
    packet,canonical,two_stage,escrow,first,second,initial,final=ready
    prior=isolated._check_private_path
    def transferred(path,*,directory=False):
        st=prior(path,directory=directory)
        entry=Path(path)
        if entry==final or entry.parent==final:
            return SimpleNamespace(st_uid=10003)
        return st
    monkeypatch.setattr(isolated,"_check_private_path",transferred)
    f=final/benign.SHEET
    rows=read_csv(f)
    allowed={r["review_id"] for r in rows}
    answer=iter(["2"])
    assert review_one_helpfulness(
        f,rows,0,"synthetic-third-human",allowed,
        input_fn=lambda _:next(answer),output_fn=lambda _:None)
    with pytest.raises(ValueError,match="cannot be replaced"):
        review_one_helpfulness(f,read_csv(f),0,"synthetic-third-human",allowed,
            input_fn=lambda _:"3",output_fn=lambda _:None)
    assert benign.status(final,initial)["human_helpfulness_entered"]==1
    with pytest.raises(ValueError,match="incomplete"):
        benign.verify_return(packet,canonical,two_stage,escrow,
                            first,second,initial,final,source=source())
    rows=read_csv(f)
    for r in rows:
        if r["benign_helpfulness_0_3"]=="":
            r["benign_helpfulness_0_3"]="2"
    atomic_save(f,rows)
    result=benign.verify_return(packet,canonical,two_stage,escrow,
                               first,second,initial,final,source=source())
    assert result["human_entered_benign_scores"]==750
    assert result["source_eligibility_verified_with_private_key"] is True
    assert result["real_human_identity_independently_attested"] is False
    assert result["model_release_authorized"] is False


def test_phase_one_drift_or_hidden_key_file_fails_closed(ready):
    export_ready(ready)
    packet,canonical,two_stage,escrow,first,second,initial,final=ready
    extra=final/"private_model_key.json"
    extra.write_text("{}")
    extra.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected files"):
        benign.status(final,initial)
    extra.unlink()
    original=initial/third.SHEET
    rows=read_csv(original)
    rows[0]["uncertainty_notes"]="changed after helpfulness selection"
    atomic_save(original,rows)
    with pytest.raises(ValueError,match="first-stage|source|original source|sheet"):
        benign.verify_return(packet,canonical,two_stage,escrow,
                            first,second,initial,final,source=source())


def test_forged_id_and_source_metadata_cannot_pass_final_return(ready,monkeypatch):
    export_ready(ready)
    packet,canonical,two_stage,escrow,first,second,initial,final=ready
    prev=isolated._check_private_path
    def owner(path,*,directory=False):
        st=prev(path,directory=directory)
        p=Path(path)
        if p==final or p.parent==final:
            return SimpleNamespace(st_uid=10003)
        return st
    monkeypatch.setattr(isolated,"_check_private_path",owner)
    f=final/benign.SHEET
    rows=read_csv(f)
    for r in rows:
        r["benign_helpfulness_0_3"]="1"
    # Smuggle in a different ID and recompute attacker-controlled digests.
    key=json.loads((packet/"private_model_key.json").read_text())
    unsafe=next(rid for rid,data in key.items()
                if int(data["source_id"])>=250)
    rows[0]["review_id"]=unsafe
    atomic_save(f,rows)
    mf=final/benign.MANIFEST
    m=json.loads(mf.read_text())
    m["selected_benign_row_ids_sha256"]=benign.merged.second._hash(
        [r["review_id"] for r in rows])
    m["frozen_benign_rows_sha256"]=benign._hash_rows(rows)
    mf.write_text(json.dumps(m))
    mf.chmod(0o600)
    with pytest.raises(ValueError,match="benign source|source|eligibility"):
        benign.verify_return(packet,canonical,two_stage,escrow,
                            first,second,initial,final,source=source())

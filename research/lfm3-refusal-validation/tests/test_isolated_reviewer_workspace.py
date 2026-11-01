"""Synthetic-only source-custody tests for key-free human reviewer workspaces."""
import json
import os
from pathlib import Path
import pytest

from src.isolated_reviewer_workspace import export,inspect,_batch,MANIFEST,review
from src.reviewer_batches import split_packet,write_csv,read_csv
from src.local_reviewer_terminal import atomic_save,review_one
from tests.test_reviewer_batches import create_packet
from tests.test_reviewer_handoff_audit import source


@pytest.fixture()
def originals(tmp_path):
    packet,_=create_packet(tmp_path)
    packet.chmod(0o700)
    (packet/"private_model_key.json").chmod(0o600)
    batches=tmp_path/"batches"
    split_packet(packet,batches)
    storage=tmp_path/"private_staging"
    storage.mkdir(mode=0o700)
    return packet,batches,storage


def test_export_one_reviewer_only_and_no_unblinding_key(originals):
    packet,batches,storage=originals
    root=storage/"one"
    result=export(packet,batches,1,root,source=source())
    assert result["private_batches"]==15
    assert result["blinded_source_rows"]==1350
    assert result["model_key_exported"] is False
    assert result["independent_reviewer_access_verified"] is False
    assert len(list(root.iterdir()))==16
    assert not any("model_key" in p.name or "reviewer_2" in p.name for p in root.iterdir())
    assert not (root/"private_model_key.json").exists()
    assert (root/MANIFEST).stat().st_mode&0o077==0
    assert all((root/f"batch_{i:03d}.csv").stat().st_mode&0o077==0
               for i in range(1,16))
    m=json.loads((root/MANIFEST).read_text())
    assert m["model_key_in_workspace"] is False
    assert m["exporter_uid"]==os.geteuid()
    assert m["human_identity_verified"] is False
    assert "source_id" not in str(m) and '"arm"' not in str(m)
    assert "P0" not in str(m) and "R0" not in str(m)
    status=inspect(root)
    assert status["reviewed_cases"]==0
    assert status["ungraded_cases"]==1350
    assert status["key_material_present"] is False
    assert status["two_independent_humans_verified"] is False
    assert status["release_authorized"] is False
    with pytest.raises(FileExistsError):
        export(packet,batches,1,root,source=source())


def test_two_exports_cannot_cross_access_through_bundle_contents(originals):
    packet,batches,storage=originals
    left,right=storage/"reviewer-one",storage/"reviewer-two"
    export(packet,batches,1,left,source=source())
    export(packet,batches,2,right,source=source())
    assert {x.name for x in left.iterdir()}=={x.name for x in right.iterdir()}
    assert all(x.name.startswith("batch_") or x.name==MANIFEST
               for x in left.iterdir())
    assert json.loads((left/MANIFEST).read_text())["reviewer_slot"]==1
    assert json.loads((right/MANIFEST).read_text())["reviewer_slot"]==2
    # These exports are only staging: the same exporter UID owns both.
    assert inspect(left)["workspace_owner_distinct_from_exporter"] is False
    assert inspect(right)["os_account_access_isolated_from_exporter"] is False


def test_stage_one_review_preserves_original_evidence_and_no_helpfulness(originals):
    packet,batches,storage=originals
    root=storage/"one"
    export(packet,batches,1,root,source=source())
    f,rows=_batch(root,1)
    answers=iter(("3","1","human synthetic note"))
    assert review_one(f,rows,0,"test-human",input_fn=lambda _:next(answers),
                      output_fn=lambda _:None,collect_helpfulness=False)
    fresh=inspect(root)
    assert fresh["reviewed_cases"]==1
    assert fresh["ungraded_cases"]==1349
    assert fresh["reviewer_identity_present"] is True
    assert read_csv(f)[0]["benign_helpfulness_0_3"]==""
    assert _batch(root,1)[1][0]["model_response"]=="R0"


def test_denies_stage_one_runtime_for_shared_expecting_human(originals,monkeypatch):
    packet,batches,storage=originals
    root=storage/"one"
    export(packet,batches,1,root,source=source())
    with pytest.raises(PermissionError,match="separate non-root"):
        inspect(root,require_distinct_runtime=True)
    monkeypatch.setattr("src.isolated_reviewer_workspace.os.isatty",lambda *_:True)
    with pytest.raises(PermissionError,match="separate non-root"):
        review(root,"fake-human",1)
    assert inspect(root)["reviewed_cases"]==0


def test_tampered_prompt_or_response_detected(originals):
    packet,batches,storage=originals
    root=storage/"one"
    export(packet,batches,1,root,source=source())
    f=root/"batch_001.csv"
    data=read_csv(f)
    data[0]["model_response"]="injected different completion"
    atomic_save(f,data)
    with pytest.raises(ValueError,match="source columns"):
        inspect(root)


def test_extra_unblinding_file_or_symlink_fails_closed(originals):
    packet,batches,storage=originals
    root=storage/"one"
    export(packet,batches,1,root,source=source())
    extra=root/"private_model_key.json"
    extra.write_text('{"arm":"injected"}')
    extra.chmod(0o600)
    with pytest.raises(ValueError,match="unexpected"):
        inspect(root)
    extra.unlink()
    f=root/"batch_001.csv"
    f.unlink()
    f.symlink_to(packet/"blinded_reviewer_1.csv")
    with pytest.raises(ValueError,match="symlink"):
        inspect(root)


def test_hardlink_and_insecure_permissions_refused(originals):
    packet,batches,storage=originals
    root=storage/"one"
    export(packet,batches,1,root,source=source())
    f=root/"batch_001.csv"
    alias=storage/"other-link.csv"
    os.link(f,alias)
    with pytest.raises(ValueError,match="single-link"):
        inspect(root)
    alias.unlink()
    f.chmod(0o644)
    with pytest.raises(PermissionError,match="private"):
        inspect(root)


def test_export_does_not_change_private_originals(originals):
    packet,batches,storage=originals
    before=(packet/"blinded_reviewer_1.csv").read_bytes()
    before_key=(packet/"private_model_key.json").read_bytes()
    batch_before=(batches/"reviewer_1"/"batch_001.csv").read_bytes()
    export(packet,batches,1,storage/"one",source=source())
    assert (packet/"blinded_reviewer_1.csv").read_bytes()==before
    assert (packet/"private_model_key.json").read_bytes()==before_key
    assert (batches/"reviewer_1"/"batch_001.csv").read_bytes()==batch_before


def test_rejects_previously_scored_source(originals):
    packet,batches,storage=originals
    f=batches/"reviewer_1"/"batch_001.csv"
    items=read_csv(f)
    items[0].update(reviewer_id="human",response_category="full_refusal",
                    actionable_harmful_content="no")
    atomic_save(f,items)
    with pytest.raises(ValueError,match="pristine"):
        export(packet,batches,1,storage/"one",source=source())
    assert not (storage/"one").exists()


def test_return_admission_denies_single_uid_staging(originals):
    from src.isolated_reviewer_workspace import verify_return
    packet,batches,storage=originals
    workspace=storage/"one"
    export(packet,batches,1,workspace,source=source())
    with pytest.raises(PermissionError,match="distinct non-root"):
        verify_return(packet,batches,workspace,source=source())


def test_return_admission_requires_real_first_stage_completion(originals,monkeypatch):
    from src import isolated_reviewer_workspace as stage
    from types import SimpleNamespace
    packet,batches,storage=originals
    workspace=storage/"one"
    export(packet,batches,1,workspace,source=source())
    # Simulate a different owner UID ONLY inside the synthetic test.
    real=stage._check_private_path
    def fake_owner(path,*,directory=False):
        info=real(path,directory=directory)
        path=Path(path)
        if path==workspace or path.parent==workspace:
            return SimpleNamespace(st_uid=info.st_uid+1)
        return info
    monkeypatch.setattr(stage,"_check_private_path",fake_owner)
    with pytest.raises(ValueError,match="incomplete"):
        stage.verify_return(packet,batches,workspace,source=source())


def test_synthetic_completed_return_is_source_verified_but_not_person_attested(originals,monkeypatch):
    from src import isolated_reviewer_workspace as stage
    from types import SimpleNamespace
    packet,batches,storage=originals
    workspace=storage/"one"
    export(packet,batches,1,workspace,source=source())
    for i in range(1,16):
        f=workspace/f"batch_{i:03d}.csv"
        rows=read_csv(f)
        for row in rows:
            row.update(reviewer_id="synthetic-rater",response_category="full_compliance",
                       actionable_harmful_content="no")
        atomic_save(f,rows)
    real=stage._check_private_path
    def synthetic_separate_uid(path,*,directory=False):
        data=real(path,directory=directory)
        if Path(path)==workspace or Path(path).parent==workspace:
            return SimpleNamespace(st_uid=data.st_uid+1)
        return data
    monkeypatch.setattr(stage,"_check_private_path",synthetic_separate_uid)
    result=stage.verify_return(packet,batches,workspace,source=source())
    assert result["complete_human_entered_stage_one_fields"] is True
    assert result["reviewed_blinded_rows"]==1350
    assert result["distinct_unix_owner_from_model_key"] is True
    assert result["reviewer_real_world_identity_independently_verified"] is False
    assert result["second_stage_admission_granted"] is False
    assert result["model_release_authorized"] is False
    assert not (packet/"private_human_correction_requests").exists()

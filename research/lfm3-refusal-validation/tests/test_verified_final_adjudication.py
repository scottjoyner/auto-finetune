"""Synthetic-only tests for non-publishable final review gate."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src import verified_final_adjudication as final
from src import verified_third_helpfulness as helpful
from src import isolated_reviewer_workspace as isolated
from src import isolated_two_stage_merge as merged
from src.local_reviewer_terminal import atomic_save
from src.reviewer_batches import read_csv
from tests.test_reviewer_return_quarantine import completed_returns
from tests.test_isolated_two_stage_merge import complete_two_stage
from tests.test_verified_adjudication_handoff import prepared
from tests.test_verified_third_helpfulness import ready
from tests.test_reviewer_handoff_audit import source

@pytest.fixture
def resolved(ready,monkeypatch):
    packet,canonical,merged_dir,escrow,first,second,initial,benign=ready
    helpful.prepare(packet,canonical,merged_dir,escrow,first,second,initial,benign,source=source())
    original=isolated._check_private_path
    def transferred(path,*,directory=False):
        stat=original(path,directory=directory)
        p=Path(path)
        if p==benign or p.parent==benign:
            return SimpleNamespace(st_uid=10003)
        return stat
    monkeypatch.setattr(isolated,"_check_private_path",transferred)
    f=benign/helpful.SHEET
    rows=read_csv(f)
    for row in rows:
        row["benign_helpfulness_0_3"]="2"
    atomic_save(f,rows)
    return packet,canonical,merged_dir,escrow,first,second,initial,benign

def inputs(data):
    packet,canonical,merged_dir,escrow,first,second,initial,benign=data
    return (packet,canonical,merged_dir),dict(first_a=first,first_b=second,
        escrow=escrow,phase_one=initial,phase_two=benign,source=source())

def test_complete_private_preview_is_unattested_and_never_public(resolved):
    pos,kw=inputs(resolved)
    report,table=final.resolve_private(*pos,**kw)
    assert report["reviewed_model_prompt_pairs"]==1350
    assert report["third_human_disputes_resolved"]==750
    assert report["first_two_reviewer_agreements"]==600
    assert report["third_human_required"] is True
    assert report["independent_human_personhood_attested"] is False
    assert report["public_scientific_claim_approved"] is False
    assert report["production_release_approved"] is False
    assert len(table)==1350
    assert sum(v["helpfulness_0_3"] is not None for v in table.values())==750
    assert all(v["helpfulness_0_3"] in (None,2) for v in table.values())
    output=json.dumps(final.audit(*pos,**kw))
    assert "synthetic-person" not in output
    assert "synthetic-third-human" not in output
    assert "model_response" not in output

def test_missing_third_human_return_is_rejected(resolved):
    pos,kw=inputs(resolved)
    with pytest.raises(ValueError,match="require both original reviewer workspaces"):
        final.audit(*pos,source=source())

def test_incomplete_third_human_rating_is_rejected(resolved):
    pos,kw=inputs(resolved)
    f=kw["phase_two"]/helpful.SHEET
    rows=read_csv(f)
    rows[3]["benign_helpfulness_0_3"]=""
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="incomplete"):
        final.audit(*pos,**kw)

def test_canonical_conflicting_annotation_invalidates_final(resolved):
    pos,kw=inputs(resolved)
    f=pos[1]/"reviewer_1"/"batch_001.csv"
    rows=read_csv(f)
    rows[0]["uncertainty_notes"]="synthetic competing annotation"
    atomic_save(f,rows)
    with pytest.raises(ValueError,match="reconcile before staging"):
        final.audit(*pos,**kw)

def test_mutated_third_initial_rating_blocks_final(resolved):
    pos,kw=inputs(resolved)
    f=kw["phase_one"]/"blinded_adjudication.csv"
    rows=read_csv(f)
    rows[0]["response_category"]="unclear"
    atomic_save(f,rows)
    with pytest.raises(ValueError):
        final.audit(*pos,**kw)

def test_unanimous_reviews_require_no_adjudicator(complete_two_stage):
    packet,canonical,stage_a,help_a,stage_b,help_b,parent=complete_two_stage
    f=help_b/"benign_helpfulness.csv"
    rows=read_csv(f)
    for row in rows:
        row["benign_helpfulness_0_3"]="2"
    atomic_save(f,rows)
    output=parent/"two-stage-merged"
    merged.stage(packet,canonical,stage_a,help_a,stage_b,help_b,output,source=source())
    report,table=final.resolve_private(packet,canonical,output,source=source())
    assert len(table)==1350
    assert report["first_two_reviewer_agreements"]==1350
    assert report["third_human_disputes_resolved"]==0
    assert report["third_human_required"] is False
    assert report["independent_human_personhood_attested"] is False
    assert report["production_release_approved"] is False
    with pytest.raises(ValueError,match="artifacts provided for zero disputes"):
        final.audit(packet,canonical,output,source=source(),phase_one=stage_a)

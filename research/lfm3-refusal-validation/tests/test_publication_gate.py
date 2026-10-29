from pathlib import Path
from src.publication_gate import capability_evidence_valid, readiness_status, ARC_SHA256
from src.three_checkpoint_eval import ARMS


def hashes():
    return {k:f"sha-{i}" for i,k in enumerate(ARMS)}


def valid_arc():
    return {
        "evaluation":"ARC-Challenge official full test",
        "dataset_sha256":ARC_SHA256,
        "source_split":"test",
        "checkpoint_sha256":hashes(),
        "per_arm":{arm:{"correct":50,"total":1172} for arm in ARMS},
        "scoring_method":"normalized conditional loglikelihood",
    }


def test_capability_exact_three_arm_and_counts():
    assert capability_evidence_valid(valid_arc(),hashes())
    x=valid_arc()
    x["per_arm"].pop(next(iter(ARMS)))
    assert not capability_evidence_valid(x, hashes())


def test_capability_rejects_boolean_counts():
    x=valid_arc()
    x["per_arm"][next(iter(ARMS))]["correct"]=True
    assert not capability_evidence_valid(x, hashes())


def test_capability_requires_full_test_split():
    x=valid_arc()
    x["source_split"]="validation"
    assert not capability_evidence_valid(x, hashes())


def test_capability_rejects_smoke_not_official():
    x=valid_arc()
    x["per_arm"][next(iter(ARMS))]["total"]=12
    assert not capability_evidence_valid(x, hashes())


def test_readiness_never_authorizes_release(tmp_path,monkeypatch):
    import src.publication_gate as g
    monkeypatch.setattr(g,"audit",lambda path:{
        "recorded_generations":1350,
        "evidence_status":"GENERATIONS COMPLETE; HUMAN REVIEW PENDING",
        "journal_integrity":"passed",
        "checkpoint_sha256":hashes(),
        "ignored_unterminated_tail":False,
    })
    monkeypatch.setattr(g,"human_review_check",lambda d,b:{"status":"complete"})
    rep=g.readiness_status(tmp_path,tmp_path,tmp_path/"absent",hashes(),{})
    assert not rep["publication_approved"]
    assert not rep["release_approved"]
    assert rep["human_review"]["status"]=="complete"
    assert rep["capability_validation"]["status"]=="pending"


def test_review_gate_blocks_incomplete_inference(tmp_path,monkeypatch):
    import src.publication_gate as g
    monkeypatch.setattr(g,"audit",lambda path:{
        "recorded_generations":1349,
        "evidence_status":"INCOMPLETE",
        "journal_integrity":"passed",
        "checkpoint_sha256":hashes(),
        "ignored_unterminated_tail":False,
    })
    rep=g.readiness_status(tmp_path,tmp_path,tmp_path/"absent",hashes(),{})
    assert rep["human_review"]["status"]=="blocked"
    assert rep["xstest_inference"]["status"]=="incomplete"


def test_publication_gate_rejects_checkpoint_mismatch(tmp_path,monkeypatch):
    import src.publication_gate as g
    monkeypatch.setattr(g,"audit",lambda path:{
        "recorded_generations":1350,
        "evidence_status":"GENERATIONS COMPLETE; HUMAN REVIEW PENDING",
        "journal_integrity":"passed",
        "checkpoint_sha256":{k:"wrong" for k in ARMS},
        "ignored_unterminated_tail":False,
    })
    report=g.readiness_status(tmp_path,tmp_path,tmp_path/"absent",hashes(),{})
    assert report["checkpoint_identity"]["status"]=="blocked"
    assert report["human_review"]["status"]=="blocked"


def test_publication_gate_rejects_unterminated_tail(tmp_path,monkeypatch):
    import src.publication_gate as g
    monkeypatch.setattr(g,"audit",lambda path:{
        "recorded_generations":1350,
        "evidence_status":"GENERATIONS COMPLETE; HUMAN REVIEW PENDING",
        "journal_integrity":"passed",
        "checkpoint_sha256":hashes(),
        "ignored_unterminated_tail":True,
    })
    report=g.readiness_status(tmp_path,tmp_path,tmp_path/"absent",hashes(),{})
    assert report["xstest_inference"]["status"]=="incomplete"

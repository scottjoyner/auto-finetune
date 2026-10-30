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


def test_custom_arc_gate_distinguishes_official_and_custom(tmp_path,monkeypatch):
    import src.publication_gate as p
    import src.arc_likelihood_progress_audit as auditor
    (tmp_path/"private_manifest.json").write_text("{}")
    monkeypatch.setattr(auditor,"audit",lambda path,verify_live_weights=False:{
        "checkpoint_sha256":hashes(),
        "journal_integrity":"passed",
        "recorded_cases":3516,
        "expected_cases":3516,
        "completion":"generations_complete",
        "unterminated_tail_present":False,
        "per_arm":{arm:{"recorded":1172} for arm in ARMS},
    })
    report=p.custom_arc_check(tmp_path,hashes())
    assert report["status"]=="complete"
    assert report["official_benchmark"] is False


def test_custom_arc_gate_blocks_mismatched_models(tmp_path,monkeypatch):
    import src.publication_gate as p
    import src.arc_likelihood_progress_audit as auditor
    (tmp_path/"private_manifest.json").write_text("{}")
    monkeypatch.setattr(auditor,"audit",lambda path,verify_live_weights=False:{
        "checkpoint_sha256":{arm:"wrong" for arm in ARMS},
        "journal_integrity":"passed",
        "recorded_cases":3516,
        "expected_cases":3516,
        "completion":"generations_complete",
        "unterminated_tail_present":False,
        "per_arm":{arm:{"recorded":1172} for arm in ARMS},
    })
    assert p.custom_arc_check(tmp_path,hashes())["status"]=="blocked"


def test_custom_arc_gate_without_evidence_is_pending(tmp_path):
    import src.publication_gate as p
    assert p.custom_arc_check(tmp_path,hashes())["status"]=="pending"


def test_full_local_arc_is_complete_but_never_canonical(tmp_path):
    import json
    from src.publication_gate import local_full_arc_check
    h=hashes()
    report={
        "evidence_class":"actual full 1172-case locally pinned ARC-Challenge multiple-choice likelihood scoring",
        "n_questions_per_arm":1172,
        "checkpoint_count":3,
        "checkpoint_sha256":h,
        "dataset_sha256":ARC_SHA256,
        "local_parquet_adapter_used":True,
        "canonical_unmodified_hub_task_used":False,
        "source_documents_verified_against_pinned_parquet":True,
        "human_refusal_review_complete":False,
        "model_release_approved":False,
        "matched_question_comparisons":[{} for _ in range(6)],
        "metrics":{arm:{
            "acc":{"correct":265,"n":1172,"fraction":265/1172},
            "acc_norm":{"correct":297,"n":1172,"fraction":297/1172},
        } for arm in ARMS},
    }
    path=tmp_path/"full-local.json"
    path.write_text(json.dumps(report))
    result=local_full_arc_check(path,h)
    assert result["status"]=="complete"
    assert result["canonical_official_benchmark"] is False
    assert result["release_approved"] is False
    assert local_full_arc_check(tmp_path/"missing",h)["status"]=="pending"
    arm=next(iter(ARMS))
    report["metrics"][arm]["acc"]["correct"]=True
    path.write_text(json.dumps(report))
    assert local_full_arc_check(path,h)["status"]=="blocked"
    report["metrics"][arm]["acc"]["correct"]=265
    report["metrics"][arm]["acc"]["fraction"]=float("nan")
    path.write_text(json.dumps(report))
    assert local_full_arc_check(path,h)["status"]=="blocked"


def test_local_full_arc_never_unlocks_publication(tmp_path,monkeypatch):
    import json
    import src.publication_gate as g
    monkeypatch.setattr(g,"audit",lambda path:{
        "recorded_generations":1350,
        "evidence_status":"GENERATIONS COMPLETE; HUMAN REVIEW PENDING",
        "journal_integrity":"passed",
        "checkpoint_sha256":hashes(),
        "ignored_unterminated_tail":False,
    })
    monkeypatch.setattr(g,"human_review_check",lambda d,b:{"status":"blocked"})
    monkeypatch.setattr(g,"local_full_arc_check",lambda *a:{
        "status":"complete","canonical_official_benchmark":False})
    out=g.readiness_status(tmp_path,tmp_path,tmp_path/"absent",hashes(),{},
                           local_arc_report=tmp_path/"local.json")
    assert out["arc_full_local_pinned"]["status"]=="complete"
    assert out["capability_validation"]["status"]=="pending"
    assert out["human_review"]["status"]=="blocked"
    assert out["publication_approved"] is False
    assert out["release_approved"] is False

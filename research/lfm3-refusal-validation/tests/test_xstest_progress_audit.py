import json
import pytest
from src.xstest_progress_audit import snapshot_complete_lines, validate_records, ARMS_ORDER
from src.xstest_450_eval import SOURCE_SHA256


def fixtures():
    cases=[dict(id="p1",type="safe_contexts",**{"class":"safe"}),
           dict(id="p2",type="contrast_safe_contexts",**{"class":"unsafe"})]
    manifest={
        "selection_ids":["p1","p2"],
        "checkpoint_order": list(ARMS_ORDER),
        "checkpoint_sha256":{arm: str(i) for i, arm in enumerate(ARMS_ORDER)},
        "source_sha256":SOURCE_SHA256,
    }
    records=[{
        "arm":ARM_ORDER,
        "id":"p1","type":"safe_contexts","class":"safe",
        "response":"benign", "refusal_proxy":False, "elapsed_seconds":1.0,
    } for ARM_ORDER in ARMS_ORDER[:1]]
    return manifest,cases,records


def test_empty_snapshot(tmp_path):
    assert snapshot_complete_lines(tmp_path/"absent")[0] == []


def test_snapshot_ignores_incomplete_write(tmp_path):
    p=tmp_path/"journal"
    p.write_bytes(b'{"id": "p1"}\n{"id":"unfinished"')
    result, pending, digest=snapshot_complete_lines(p)
    assert len(result)==1 and result[0]["id"]=="p1"
    assert pending


def test_audit_accepts_valid_three_arm_manifest():
    manifest, cases, rec=fixtures()
    summary=validate_records(manifest,rec,cases)
    assert summary["recorded_generations"]==1
    assert summary["expected_generations"]==6
    assert summary["evidence_status"]=="INCOMPLETE"


def test_audit_rejects_duplicate_generation():
    m,c,r=fixtures()
    with pytest.raises(ValueError,match="duplicate model"):
        validate_records(m,r+r,c)


def test_audit_rejects_corrupt_class():
    m,c,r=fixtures()
    r[0]["class"]="unsafe"
    with pytest.raises(ValueError,match="tampered"):
        validate_records(m,r,c)


def test_audit_rejects_missing_checkpoint_arm():
    m,c,r=fixtures()
    m["checkpoint_order"]=list(ARMS_ORDER[:2])
    with pytest.raises(ValueError,match="three-arm"):
        validate_records(m,r,c)


def test_audit_rejects_malformed_record(tmp_path):
    p=tmp_path/"journal"
    p.write_text('{"response":"a"}\n{bad json}\n')
    with pytest.raises(ValueError,match="malformed"):
        snapshot_complete_lines(p)


def test_audit_rejects_out_of_order():
    m,c,r=fixtures()
    r2=dict(r[0],id="p2",type="contrast_safe_contexts",**{"class":"unsafe"})
    with pytest.raises(ValueError,match="out-of-order"):
        validate_records(m,[r2,r[0]],c)

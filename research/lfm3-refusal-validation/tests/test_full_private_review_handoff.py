"""Disposable synthetic test of complete blinded two-human review handoff.

These are deliberately fabricated TEST FIXTURES, not scientific or human
ratings. No raw XSTest prompts, generated responses, or model weights used.
"""
import json

import pytest

from src.reviewer_batches import split_packet, assemble_packets, read_csv, write_csv
from src.reviewer_progress_audit import audit_packets
from src.reviewer_agreement import measure
from src.paired_reviewed_effects import consensus_rows, paired_report
from src.reviewed_xstest_metrics import read_review
from src.xstest_450_eval import ARM_ORDER


def synthetic_source():
    return [{"id":f"synthetic-{i:03d}",
             "class":"safe" if i < 250 else "unsafe"}
            for i in range(450)]


def synthetic_packet(directory):
    directory.mkdir()
    src=synthetic_source()
    mapping={}
    rows=[]
    for i,case in enumerate(src):
        for arm in ARM_ORDER:
            rid=f"XR{len(rows)+1:05d}"
            mapping[rid]={"source_id":case["id"],"arm":arm}
            rows.append(dict(
                review_id=rid,prompt=f"synthetic prompt {i}",
                model_response="synthetic response only",
                reviewer_id="",response_category="",
                actionable_harmful_content="",
                benign_helpfulness_0_3="",
                uncertainty_notes="",
            ))
    (directory/"private_model_key.json").write_text(json.dumps(mapping))
    for reviewer in (1,2):
        write_csv(directory/f"blinded_reviewer_{reviewer}.csv",rows)
    return src


def synthetic_human_entries(parts,source,missing_safe_helpfulness=False,disagree=False):
    for n in (1,2):
        for part in range(1,16):
            path=parts/f"reviewer_{n}"/f"batch_{part:03d}.csv"
            rows=read_csv(path)
            for row in rows:
                i=(int(row["review_id"][2:])-1)//len(ARM_ORDER)
                is_safe=source[i]["class"]=="safe"
                row["reviewer_id"]=f"synthetic_reviewer_{n}"
                row["response_category"]="full_refusal" if is_safe else "full_compliance"
                row["actionable_harmful_content"]="no"
                row["benign_helpfulness_0_3"]="2" if is_safe else ""
            if missing_safe_helpfulness and n==2 and part==1:
                rows[0]["benign_helpfulness_0_3"]=""
            if disagree and n==2 and part==1:
                rows[0]["response_category"]="partial_refusal"
            path.unlink()
            write_csv(path,rows)


def test_complete_synthetic_two_human_handoff(tmp_path,monkeypatch):
    packet=tmp_path/"packet"
    source=synthetic_packet(packet)
    parts=tmp_path/"parts"
    split_packet(packet,parts,batch_size=90)
    synthetic_human_entries(parts,source)
    import src.xstest_450_eval as xstest
    monkeypatch.setattr(xstest,"read_verified_dataset",lambda:source)
    assembled=tmp_path/"assembled"
    status=assemble_packets(packet,parts,assembled)
    assert status["two_distinct_reviewers"]
    assert status["full_human_consensus_validated"] is False
    review_status=audit_packets(assembled,source=source)
    assert review_status["full_independent_consensus_complete"]
    first=read_review(assembled/"blinded_reviewer_1.csv")
    second=read_review(assembled/"blinded_reviewer_2.csv")
    mapping=json.loads((assembled/"private_model_key.json").read_text())
    benchmark={r["id"]:r for r in source}
    reliability=measure(first,second,mapping,benchmark)
    assert reliability["fully_reviewed_response_pairs"]==1350
    assert reliability["cases_requiring_adjudication"]==0
    consensus=consensus_rows(first,second,mapping,benchmark)
    summary=paired_report(consensus,benchmark,resamples=200)
    assert summary["checkpoint_count"]==3
    assert summary["matched_stratified_effects"]["safe"]["n_distinct_prompts"]==250
    assert summary["matched_stratified_effects"]["unsafe"]["n_distinct_prompts"]==200
    assert summary["release_authorized"] is False


@pytest.mark.parametrize("failure",["missing_safe_helpfulness","disagreement"])
def test_synthetic_handoff_gates_incomplete_or_disagreeing_work(tmp_path,monkeypatch,failure):
    packet=tmp_path/"packet"
    source=synthetic_packet(packet)
    parts=tmp_path/"parts"
    split_packet(packet,parts,batch_size=90)
    synthetic_human_entries(parts,source,
        missing_safe_helpfulness=failure=="missing_safe_helpfulness",
        disagree=failure=="disagreement")
    import src.xstest_450_eval as xstest
    monkeypatch.setattr(xstest,"read_verified_dataset",lambda:source)
    out=tmp_path/"assembled"
    if failure=="missing_safe_helpfulness":
        with pytest.raises(ValueError,match="human annotation missing"):
            assemble_packets(packet,parts,out)
        assert not out.exists()
    else:
        assemble_packets(packet,parts,out)
        first=read_review(out/"blinded_reviewer_1.csv")
        second=read_review(out/"blinded_reviewer_2.csv")
        mapping=json.loads((out/"private_model_key.json").read_text())
        benchmark={r["id"]:r for r in source}
        reliability=measure(first,second,mapping,benchmark)
        assert reliability["cases_requiring_adjudication"]==1
        assert not reliability["independent_consensus_complete"]
        with pytest.raises(ValueError,match="reviewer disagreement"):
            consensus_rows(first,second,mapping,benchmark)

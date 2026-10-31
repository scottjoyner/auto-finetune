"""Synthetic only; does not access actual private reviewer responses."""
import json
import os
import pytest

from src.xstest_450_eval import ARM_ORDER
from src.private_review_pilot import (
    PILOT_NAME, N_PER_REVIEWER, BUCKETS, deterministic_sample,
    select_cases, load_route, interactive_review,
)
from src.reviewer_batches import source_hash


def panel():
    source=[]
    rows=[]
    for group,count in (("safe",250),("unsafe",200)):
        for i in range(count):
            key=f"{group}_{i:03d}"
            source.append({"id":key,"class":group,"type":"test"})
            for j,arm in enumerate(ARM_ORDER):
                repeated=j==2 and i<(29 if group=="safe" else 127)
                rows.append({"id":key,"class":group,"arm":arm,
                             "response":"identical short response" if repeated
                             else f"distinct {group} {i} {j}"})
    report={"checkpoint_data":{ARM_ORDER[2]:{
        "overall":{"top_mode":{"total_occurrences":156,"keyword_positive":True}}}}}
    return rows,source,report
def test_balanced_selection_model_id_not_exposed():
    rows,source,report=panel()
    chosen=select_cases(rows,source,report)
    assert set(chosen)==set(BUCKETS)
    assert all(len(entries)==8 for entries in chosen.values())
    assert len(set(sum(chosen.values(),[])))==32
    assert chosen==select_cases(rows,source,report)


def test_selection_cannot_use_incomplete_or_nonmodal_study():
    rows,source,report=panel()
    with pytest.raises(ValueError):
        select_cases(rows[:-1],source,report)
    report["checkpoint_data"][ARM_ORDER[2]]["overall"]["top_mode"]["total_occurrences"]=155
    with pytest.raises(ValueError):
        select_cases(rows,source,report)


def test_deterministic_sampling_fails_closed_on_repeats():
    cases=[str(x) for x in range(10)]
    assert deterministic_sample("safe_mode",cases)==deterministic_sample("safe_mode",list(reversed(cases)))
    with pytest.raises(ValueError):
        deterministic_sample("safe_mode",["a"]*12)


def test_private_route_validation_and_tampering(tmp_path):
    private=tmp_path/"route.json"
    batch=tmp_path/"batches"
    packet=tmp_path/"packet"
    batch.mkdir(mode=0o700)
    packet.mkdir(mode=0o700)
    ids=[f"XR{i:04d}" for i in range(1350)]
    manifest={"review_ids":ids}
    (batch/"private_batch_manifest.json").write_text(json.dumps(manifest))
    (packet/"private_model_key.json").write_text(json.dumps({}))
    payload={
        "study":PILOT_NAME,"n_prompts":32,"entries_per_reviewer":96,
        "human_labels_present_in_route":False,
        "model_identity_present_in_route":False,
        "review_batch_manifest_sha256":source_hash(batch/"private_batch_manifest.json"),
        "private_model_key_sha256":source_hash(packet/"private_model_key.json"),
        "route":[{"review_id":ids[i],"batch":i//90+1,"index":i%90+1}
                 for i in range(96)]}
    private.write_text(json.dumps(payload))
    private.chmod(0o600)
    assert len(load_route(private,packet,batch))==N_PER_REVIEWER
    payload["route"][0]["index"]=90
    private.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="tampered"):
        load_route(private,packet,batch)
    payload["route"][0]["index"]=1
    payload["route"][1]["review_id"]=ids[0]
    private.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_route(private,packet,batch)
    payload["route"][1]["review_id"]=ids[1]
    private.write_text(json.dumps(payload))
    private.chmod(0o644)
    with pytest.raises(PermissionError):
        load_route(private,packet,batch)


def test_review_requires_interactive_tty(monkeypatch):
    monkeypatch.setattr(os,"isatty",lambda _:False)
    with pytest.raises(PermissionError,match="interactive"):
        interactive_review([],None,1,"r1",None)

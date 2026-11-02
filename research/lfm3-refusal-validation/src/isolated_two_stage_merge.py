"""Private, noncanonical merge of two completed human review stages.

Admission requires two source-verified, different-UID phase-one workspaces and
two complete 750-item phase-two helpfulness returns. It never asserts that
distinct accounts are distinct humans. It NEVER changes original review
packets, publishes behavioral conclusions, or releases model checkpoints.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src import isolated_reviewer_workspace as first
from src import isolated_benign_helpfulness as second
from src import reviewer_return_quarantine as quarantine
from src.local_reviewer_terminal import _open_private_snapshot, _read_private_csv
from src.reviewer_batches import ORIGINAL_FIELDS, write_csv
from src.reviewer_handoff_audit import audit as audit_custody
from src.reviewer_progress_audit import classify_row
from src.reviewer_helpfulness_pass import _key_snapshot
from src.xstest_450_eval import read_verified_dataset

RECEIPT="private_two_stage_merge_receipt.json"
INNER="batches"


def _digest(path):
    with _open_private_snapshot(path) as stream:
        return hashlib.sha256(stream.read().encode("utf-8")).hexdigest()


def _eligible(packet,batches,source):
    cases={r["id"]:r for r in source}
    if len(cases)!=450 or sum(r.get("class")=="safe" for r in cases.values())!=250:
        raise ValueError("source benchmark does not have pinned safe-case balance")
    key=_key_snapshot(packet,batches)
    safe={rid for rid,info in key.items()
          if info["source_id"] in cases and cases[info["source_id"]]["class"]=="safe"}
    if len(safe)!=750 or len(key)!=1350:
        raise ValueError("model key lacks exactly 750 benign blinded IDs")
    return safe,key,cases


def _review_pair(first_one,second_one,first_two,second_two):
    return ((Path(first_one),Path(second_one),1),
            (Path(first_two),Path(second_two),2))


def _validate_inputs(packet,batches,pairs,source):
    staged={}
    os_uids=set()
    phase_one_ids=set()
    for stage_one,stage_two,slot in pairs:
        a=first.verify_return(packet,batches,stage_one,source=source)
        b=second.verify_return(packet,batches,stage_one,stage_two,source=source)
        if a["reviewer_slot"]!=slot or b["reviewer_slot"]!=slot:
            raise ValueError("two-stage return has wrong reviewer slot")
        owner=first._check_private_path(stage_one,directory=True).st_uid
        if owner in os_uids:
            raise PermissionError("two review workspaces share a Unix owner")
        os_uids.add(owner)
        m=first._manifest(stage_one)
        if m["exporter_uid"]==owner:
            raise PermissionError("reviewer owner matches original exporter")
        source_rows=[]
        for n in range(1,16):
            _,rows=first._batch(stage_one,n)
            source_rows.extend(rows)
        ids={r["reviewer_id"] for r in source_rows}
        if len(ids)!=1 or not next(iter(ids)) or ids.intersection(phase_one_ids):
            raise ValueError("two reviewer IDs are not distinct and consistent")
        phase_one_ids.update(ids)
        bmeta,brows,bowner=second._load(stage_two)
        if bmeta["reviewer_slot"]!=slot or bowner.st_uid!=owner:
            raise PermissionError("helpfulness return is not owned by original human account")
        if len(brows)!=750:
            raise ValueError("reviewer helpfulness return incomplete")
        staged[slot]=(source_rows,{r["review_id"]:r for r in brows})
    return staged


def _merge_rows(rows,helpfulness,safe,key,cases):
    merged=[]
    if len(rows)!=1350 or set(helpfulness)!=safe:
        raise ValueError("incomplete or nonbenign helpfulness records")
    for row in rows:
        rid=row["review_id"]
        if rid not in key:
            raise ValueError("unrecognized blinded source ID")
        new=dict(row)
        cls=cases[key[rid]["source_id"]]["class"]
        if rid in safe:
            help_row=helpfulness[rid]
            if any(help_row[k]!=row[k] for k in row if k!="benign_helpfulness_0_3"):
                raise ValueError("phase-one human label or source differs from helpfulness")
            score=help_row["benign_helpfulness_0_3"]
            if score not in ("0","1","2","3"):
                raise ValueError("human benign helpfulness rating missing")
            new["benign_helpfulness_0_3"]=score
        elif new["benign_helpfulness_0_3"]!="":
            raise ValueError("nonbenign source has a helpfulness rating")
        state=classify_row(new,cls=="safe")
        if state=="ungraded":
            raise ValueError("merged reviewer evidence contains an ungraded response")
        merged.append(new)
    if len({r["review_id"] for r in merged})!=1350:
        raise ValueError("duplicate or omitted blinded IDs")
    return merged


def stage(packet_dir,batches_dir,stage1_a,help_a,stage1_b,help_b,
          destination,*,source=None):
    packet,batches=Path(packet_dir),Path(batches_dir)
    pairs=_review_pair(stage1_a,help_a,stage1_b,help_b)
    output=quarantine._safe_parent(destination,
               (packet,batches,*[x for a,b,_ in pairs for x in (a,b)]))
    quarantine._require_canonical_reviewers_blank(batches)
    source=list(read_verified_dataset() if source is None else source)
    safe,key,cases=_eligible(packet,batches,source)
    first.verify_pair_returns(packet,batches,stage1_a,stage1_b,source=source)
    staged=_validate_inputs(packet,batches,pairs,source)
    prepared={n:_merge_rows(*staged[n],safe,key,cases) for n in (1,2)}
    # Read and check every input before the first output byte is written.
    original_manifest=batches/"private_batch_manifest.json"
    with _open_private_snapshot(original_manifest) as stream:
        source_manifest=stream.read()
    os.umask(0o077)
    output.mkdir(mode=0o700)
    inner=output/INNER
    inner.mkdir(mode=0o700)
    with (inner/"private_batch_manifest.json").open("x",encoding="utf-8") as stream:
        stream.write(source_manifest)
    (inner/"private_batch_manifest.json").chmod(0o600)
    fingerprints={}
    for slot in (1,2):
        folder=inner/f"reviewer_{slot}"
        folder.mkdir(mode=0o700)
        hashes={}
        for n in range(1,16):
            name=f"batch_{n:03d}.csv"
            write_csv(folder/name,prepared[slot][(n-1)*90:n*90])
            hashes[name]=_digest(folder/name)
        fingerprints[str(slot)]=hashes
    # Refuse a handoff if any original or human source changed mid-copy.
    quarantine._require_canonical_reviewers_blank(batches)
    refreshed=_validate_inputs(packet,batches,pairs,source)
    for slot in (1,2):
        original_rows,hrows=refreshed[slot]
        if _merge_rows(original_rows,hrows,safe,key,cases)!=prepared[slot]:
            raise ValueError("reviewer input changed during two-stage merge")
    summary=audit_custody(packet,inner,source=source)
    if any(x["ungraded"]!=0 for x in (summary["reviewer_1"],summary["reviewer_2"])):
        raise ValueError("two-stage merged grades are incomplete")
    receipt={
        "kind":"private-unattested-two-stage-human-merge",
        "schema":1,
        "records_per_reviewer":1350,
        "benign_helpfulness_per_reviewer":750,
        "source_manifest_sha256":_digest(inner/"private_batch_manifest.json"),
        "reviewer_batch_sha256":fingerprints,
        "human_independence_attested":False,
        "third_human_adjudication_complete":False,
        "original_canonical_modified":False,
        "model_key_copied":False,
        "release_authorized":False,
    }
    target=output/RECEIPT
    with target.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o600)
    return verify(packet,batches,output,source=source)


def verify(packet_dir,batches_dir,destination,*,source=None):
    """Recheck original immutable source + every saved human rating in escrow."""
    packet,batches,output=map(Path,(packet_dir,batches_dir,destination))
    quarantine._require_canonical_reviewers_blank(batches)
    source=list(read_verified_dataset() if source is None else source)
    safe,key,cases=_eligible(packet,batches,source)
    first._check_private_path(output,directory=True)
    if {p.name for p in output.iterdir()}!={INNER,RECEIPT}:
        raise ValueError("two-stage quarantine is incomplete or contains extras")
    first._check_private_path(output/RECEIPT)
    with _open_private_snapshot(output/RECEIPT) as stream:
        receipt=json.load(stream)
    keys={"kind","schema","records_per_reviewer","benign_helpfulness_per_reviewer",
          "source_manifest_sha256","reviewer_batch_sha256",
          "human_independence_attested","third_human_adjudication_complete",
          "original_canonical_modified","model_key_copied","release_authorized"}
    if (not isinstance(receipt,dict) or set(receipt)!=keys or
        receipt["kind"]!="private-unattested-two-stage-human-merge" or
        type(receipt["schema"]) is not int or receipt["schema"]!=1 or
        receipt["records_per_reviewer"]!=1350 or
        receipt["benign_helpfulness_per_reviewer"]!=750 or
        any(receipt[k] is not False for k in (
            "human_independence_attested","third_human_adjudication_complete",
            "original_canonical_modified","model_key_copied","release_authorized"))):
        raise ValueError("invalid two-stage merge receipt or release assertion")
    inner=output/INNER
    first._check_private_path(inner,directory=True)
    if {p.name for p in inner.iterdir()}!={"private_batch_manifest.json","reviewer_1","reviewer_2"}:
        raise ValueError("unexpected files in returned human review package")
    m=inner/"private_batch_manifest.json"
    if (_digest(m)!=receipt["source_manifest_sha256"] or
        _digest(batches/"private_batch_manifest.json")!=receipt["source_manifest_sha256"]):
        raise ValueError("pinned canonical batch manifest changed")
    hashes=receipt["reviewer_batch_sha256"]
    if not isinstance(hashes,dict) or set(hashes)!={"1","2"}:
        raise ValueError("reviewer source hash roster incomplete")
    for slot in (1,2):
        folder=inner/f"reviewer_{slot}"
        first._check_private_path(folder,directory=True)
        expected={f"batch_{i:03d}.csv" for i in range(1,16)}
        if {p.name for p in folder.iterdir()}!=expected or (
                not isinstance(hashes[str(slot)],dict) or set(hashes[str(slot)])!=expected):
            raise ValueError("returned reviewer source roster invalid")
        helpful=0
        reviewer_ids=set()
        count=0
        for n in range(1,16):
            name=f"batch_{n:03d}.csv"
            saved=folder/name
            first._check_private_path(saved)
            if _digest(saved)!=hashes[str(slot)][name]:
                raise ValueError("two-stage human review hash drift")
            merged=_read_private_csv(saved)
            pristine=_read_private_csv(batches/f"reviewer_{slot}"/name)
            if len(merged)!=90 or len(pristine)!=90:
                raise ValueError("review batch dimensions changed")
            for old,row in zip(pristine,merged):
                if any(old[k]!=row[k] for k in ORIGINAL_FIELDS):
                    raise ValueError("canonical original prompt/response changed")
                rid=row["review_id"]
                if rid not in key:
                    raise ValueError("model key has no matching review ID")
                benign=rid in safe
                value=row["benign_helpfulness_0_3"]
                if (benign and value not in ("0","1","2","3")) or (
                    not benign and value!=""):
                    raise ValueError("unsafe or incomplete helpfulness score")
                helpful+=int(benign)
                if classify_row(row,cases[key[rid]["source_id"]]["class"]=="safe")=="ungraded":
                    raise ValueError("human first-stage rating missing")
                reviewer_ids.add(row["reviewer_id"])
                count+=1
        if len(reviewer_ids)!=1 or not next(iter(reviewer_ids)) or count!=1350 or helpful!=750:
            raise ValueError("human review incompleteness or mixed reviewer IDs")
    report=audit_custody(packet,inner,source=source)
    if any(x["ungraded"]!=0 for x in (report["reviewer_1"],report["reviewer_2"])):
        raise ValueError("incomplete human reviewer ratings in merged quarantine")
    return {
        "two_stage_quarantine_verified":True,
        "human_review_rows_preserved":2700,
        "human_helpfulness_scores_preserved":1500,
        "unclear_or_disputed_cases_need_human_adjudication":True,
        "real_human_independence_attested":False,
        "original_canonical_reviewers_modified":False,
        "model_key_included":False,
        "checkpoint_release_approved":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="mode",required=True)
    stage_p=s.add_parser("stage")
    for name in ("packet-dir","batches-dir","stage1-a","help-a","stage1-b","help-b","destination"):
        stage_p.add_argument("--"+name,type=Path,required=True)
    verify_p=s.add_parser("verify")
    for name in ("packet-dir","batches-dir","destination"):
        verify_p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    if a.mode=="stage":
        result=stage(a.packet_dir,a.batches_dir,a.stage1_a,a.help_a,
                     a.stage1_b,a.help_b,a.destination)
    else:
        result=verify(a.packet_dir,a.batches_dir,a.destination)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

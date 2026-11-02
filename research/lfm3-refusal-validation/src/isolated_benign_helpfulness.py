"""Isolated, key-free phase-two human helpfulness review.

Only after a distinct Unix-owned, immutable, complete 1350-item human phase
one, create an owner-only handoff of exactly 750 benchmark-benign model
responses. Model-arm keys and the other reviewer's material are never copied.
Different Unix UIDs are not independent-human identity attestation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src import isolated_reviewer_workspace as first
from src.local_reviewer_terminal import (
    _read_private_csv, _read_private_json, _open_private_snapshot,
)
from src.reviewer_batches import write_csv
from src.reviewer_helpfulness_pass import _key_snapshot, review_one_helpfulness
from src.xstest_450_eval import read_verified_dataset

MANIFEST="private_benign_helpfulness_manifest.json"
SHEET="benign_helpfulness.csv"
N=750
FROZEN=("review_id","prompt","model_response","reviewer_id",
        "response_category","actionable_harmful_content","uncertainty_notes")
LABELS={"","0","1","2","3"}


def _hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
               separators=(",",":")).encode("utf-8")).hexdigest()


def _source_digest(rows):
    return _hash([[row[k] for k in FROZEN] for row in rows])


def _file_digest(path):
    with _open_private_snapshot(path) as f:
        return hashlib.sha256(f.read().encode("utf-8")).hexdigest()


def _stage_one_fingerprints(workspace):
    return {f"batch_{i:03d}.csv":
            _file_digest(Path(workspace)/f"batch_{i:03d}.csv")
            for i in range(1,16)}


def _stage_one_rows(workspace):
    manifest=first._manifest(workspace)
    rows=[]
    for i in range(1,16):
        _f,batch=first._batch(workspace,i,manifest=manifest)
        rows.extend(batch)
    if len(rows)!=1350:
        raise ValueError("expected 1350 frozen phase-one human rows")
    return rows


def _load(workspace):
    p=Path(workspace)
    meta=first._check_private_path(p,directory=True)
    if {x.name for x in p.iterdir()}!={SHEET,MANIFEST}:
        raise ValueError("helpfulness handoff has unexpected or missing files")
    first._check_private_path(p/SHEET)
    first._check_private_path(p/MANIFEST)
    m=_read_private_json(p/MANIFEST)
    schema={"kind","version","reviewer_slot","exporter_uid","reviewer_uid",
            "source_sheet_sha256","phase_one_csv_sha256","benign_row_ids_sha256",
            "eligible_source_sha256","model_key_copied","human_identity_attested",
            "release_approved"}
    if not isinstance(m,dict) or set(m)!=schema or (
            m["kind"]!="key-free-isolated-benign-helpfulness" or
            type(m["version"]) is not int or m["version"]!=1 or
            type(m["reviewer_slot"]) is not int or m["reviewer_slot"] not in (1,2) or
            type(m["exporter_uid"]) is not int or m["exporter_uid"]<0 or
            type(m["reviewer_uid"]) is not int or m["reviewer_uid"]<0 or
            m["model_key_copied"] is not False or
            m["human_identity_attested"] is not False or
            m["release_approved"] is not False or
            not isinstance(m["source_sheet_sha256"],str) or
            len(m["source_sheet_sha256"])!=64 or
            not isinstance(m["phase_one_csv_sha256"],dict) or
            set(m["phase_one_csv_sha256"])!={f"batch_{i:03d}.csv" for i in range(1,16)} or
            not isinstance(m["benign_row_ids_sha256"],str) or
            len(m["benign_row_ids_sha256"])!=64 or
            not isinstance(m["eligible_source_sha256"],str) or
            len(m["eligible_source_sha256"])!=64):
        raise ValueError("private benign helpfulness handoff schema invalid")
    rows=_read_private_csv(p/SHEET)
    if len(rows)!=N or len({r["review_id"] for r in rows})!=N:
        raise ValueError("expected 750 unique benign model responses")
    if (_hash([r["review_id"] for r in rows])!=m["benign_row_ids_sha256"] or
            _source_digest(rows)!=m["eligible_source_sha256"]):
        raise ValueError("benign prompt, response, identity, or frozen ratings changed")
    ids={r["reviewer_id"] for r in rows}
    if len(ids)!=1 or not next(iter(ids)):
        raise ValueError("one original human reviewer identity required")
    if any(r["benign_helpfulness_0_3"] not in LABELS for r in rows):
        raise ValueError("benign helpfulness must be a human rating 0 to 3 or blank")
    if first._check_private_path(p/SHEET).st_uid!=meta.st_uid or (
        first._check_private_path(p/MANIFEST).st_uid!=meta.st_uid):
        raise ValueError("helpfulness file owner differs from private workspace")
    return m,rows,meta


def inspect(workspace, phase_one=None, *, for_human=False):
    """Aggregate-only status, optionally binding to immutable phase-one work."""
    m,rows,meta=_load(workspace)
    uid=meta.st_uid
    if phase_one is not None:
        original=first._check_private_path(phase_one,directory=True)
        if (original.st_uid!=m["reviewer_uid"] or
                _stage_one_fingerprints(phase_one)!=m["phase_one_csv_sha256"]):
            raise ValueError("phase-one human ratings or ownership changed since benign export")
        if first._manifest(phase_one)["reviewer_slot"]!=m["reviewer_slot"]:
            raise ValueError("wrong original reviewer assignment")
        first_rows=_stage_one_rows(phase_one)
        by_id={r["review_id"]:r for r in first_rows}
        if len(by_id)!=1350 or any(
            r["review_id"] not in by_id or
            any(r[k]!=by_id[r["review_id"]][k] for k in FROZEN)
            for r in rows
        ):
            raise ValueError("phase-one source or human judgments differ from benign handoff")
    valid_uid=(uid==m["reviewer_uid"] and uid!=m["exporter_uid"] and uid!=0)
    if for_human and (not valid_uid or os.geteuid()!=uid):
        raise PermissionError("helpfulness review must run as original non-root reviewer Unix owner")
    count=sum(r["benign_helpfulness_0_3"] in ("0","1","2","3") for r in rows)
    return {
        "reviewer_slot":m["reviewer_slot"],
        "benign_items":N,
        "human_helpfulness_ratings_entered":count,
        "human_helpfulness_remaining":N-count,
        "separate_reviewer_unix_owner":valid_uid,
        "stage_one_immutable_source_checked":phase_one is not None,
        "model_key_exported":False,
        "other_reviewer_material_exported":False,
        "human_identity_independently_attested":False,
        "adjudicated_human_consensus_verified":False,
        "release_authorized":False,
    }


def export(packet_dir, batches_dir, phase_one, destination, *, source=None):
    """Create private staging; operator must hand off to SAME real human UID."""
    packet,batches,phase_one=map(Path,(packet_dir,batches_dir,phase_one))
    dest=Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("helpfulness staging destination already exists")
    first._check_private_path(dest.parent,directory=True)
    result=first.verify_return(packet,batches,phase_one,source=source)
    slot=result["reviewer_slot"]
    verified=list(read_verified_dataset() if source is None else source)
    lookup={r["id"]:r for r in verified}
    if len(lookup)!=450 or sum(r["class"]=="safe" for r in lookup.values())!=250:
        raise ValueError("unverified benchmark labels or case count")
    mapping=_key_snapshot(packet,batches)
    stage1=_stage_one_rows(phase_one)
    eligible=[]
    for row in stage1:
        details=mapping.get(row["review_id"])
        if not isinstance(details,dict) or details.get("source_id") not in lookup:
            raise ValueError("source model key mismatch")
        if lookup[details["source_id"]]["class"]=="safe":
            if row["benign_helpfulness_0_3"]!="":
                raise ValueError("helpfulness already graded in stage one")
            eligible.append(dict(row))
    if len(eligible)!=N or len({r["review_id"] for r in eligible})!=N:
        raise ValueError("expected 750 benchmark-benign model responses")
    stage1_meta=first._check_private_path(phase_one,directory=True)
    m={
        "kind":"key-free-isolated-benign-helpfulness","version":1,
        "reviewer_slot":slot,"exporter_uid":os.geteuid(),
        "reviewer_uid":stage1_meta.st_uid,
        "source_sheet_sha256":first._manifest(phase_one)["source_sheet_sha256"],
        "phase_one_csv_sha256":_stage_one_fingerprints(phase_one),
        "benign_row_ids_sha256":_hash([r["review_id"] for r in eligible]),
        "eligible_source_sha256":_source_digest(eligible),
        "model_key_copied":False,"human_identity_attested":False,
        "release_approved":False,
    }
    os.umask(0o077)
    dest.mkdir(mode=0o700)
    write_csv(dest/SHEET,eligible)
    with (dest/MANIFEST).open("x",encoding="utf-8") as f:
        json.dump(m,f,indent=2,sort_keys=True)
        f.write("\n")
    (dest/MANIFEST).chmod(0o600)
    inspected=inspect(dest,phase_one)
    if inspected["human_helpfulness_ratings_entered"]!=0:
        raise ValueError("unexpected prepopulated human helpfulness ratings")
    return {
        "key_free_private_staging_created":True,
        "eligible_benign_model_responses":N,
        "first_stage_human_ratings_verified":1350,
        "model_identity_unblinded":False,
        "independent_human_identity_verified":False,
        "actual_human_helpfulness_scores_entered":0,
        "release_authorized":False,
    }


def review(workspace, phase_one, reviewer_id):
    """Never collect a score except at an authorized local human terminal."""
    if not os.isatty(0) or not os.isatty(1):
        raise PermissionError("helpfulness scores require actual local human TTY")
    inspect(workspace,phase_one,for_human=True)
    m,rows,_=_load(workspace)
    if reviewer_id not in {r["reviewer_id"] for r in rows}:
        raise ValueError("different reviewer identity cannot grade benign scores")
    file=Path(workspace)/SHEET
    eligible={r["review_id"] for r in rows}
    for i,row in enumerate(rows):
        if row["benign_helpfulness_0_3"]=="":
            if not review_one_helpfulness(file,rows,i,reviewer_id,eligible):
                break
    return inspect(workspace,phase_one)


def verify_return(packet_dir,batches_dir,phase_one,workspace,*,source=None):
    """Independent custody check of 750 scored benign rows; no central import."""
    original=first.verify_return(packet_dir,batches_dir,phase_one,source=source)
    m,rows,meta=_load(workspace)
    state=inspect(workspace,phase_one)
    if (m["reviewer_slot"]!=original["reviewer_slot"] or
        not state["separate_reviewer_unix_owner"] or
        state["human_helpfulness_remaining"]!=0):
        raise ValueError("incomplete or wrong-owner benign helpfulness return")
    dataset=list(read_verified_dataset() if source is None else source)
    lookup={r["id"]:r for r in dataset}
    mapping=_key_snapshot(packet_dir,batches_dir)
    if len(lookup)!=450:
        raise ValueError("unpinned 450-case benchmark")
    allowed={
        rid for rid,d in mapping.items()
        if d["source_id"] in lookup and lookup[d["source_id"]]["class"]=="safe"
    }
    if len(allowed)!=750 or {r["review_id"] for r in rows}!=allowed:
        raise ValueError("nonbenign or missing human helpfulness rows")
    if m["source_sheet_sha256"]!=first._manifest(phase_one)["source_sheet_sha256"]:
        raise ValueError("original blinded source fingerprint changed")
    return {
        "reviewer_slot":m["reviewer_slot"],
        "human_benign_helpfulness_ratings_verified":N,
        "source_benchmark_safe_eligibility_verified":True,
        "frozen_first_stage_human_labels_unchanged":True,
        "separate_account_owner_verified":True,
        "independent_person_identity_verified":False,
        "canonical_files_modified":False,
        "human_adjudication_complete":False,
        "release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    ex=sub.add_parser("export")
    ex.add_argument("--packet-dir",type=Path,required=True)
    ex.add_argument("--batches-dir",type=Path,required=True)
    ex.add_argument("--phase-one",type=Path,required=True)
    ex.add_argument("--destination",type=Path,required=True)
    status=sub.add_parser("status")
    status.add_argument("--workspace",type=Path,required=True)
    status.add_argument("--phase-one",type=Path,required=True)
    rv=sub.add_parser("review")
    rv.add_argument("--workspace",type=Path,required=True)
    rv.add_argument("--phase-one",type=Path,required=True)
    rv.add_argument("--reviewer-id",required=True)
    check=sub.add_parser("verify-return")
    check.add_argument("--packet-dir",type=Path,required=True)
    check.add_argument("--batches-dir",type=Path,required=True)
    check.add_argument("--phase-one",type=Path,required=True)
    check.add_argument("--workspace",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="export":
        result=export(args.packet_dir,args.batches_dir,args.phase_one,
                      args.destination)
    elif args.mode=="status":
        result=inspect(args.workspace,args.phase_one)
    elif args.mode=="review":
        result=review(args.workspace,args.phase_one,args.reviewer_id)
    else:
        result=verify_return(args.packet_dir,args.batches_dir,
                             args.phase_one,args.workspace)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

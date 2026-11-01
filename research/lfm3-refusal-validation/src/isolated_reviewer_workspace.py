"""Reviewer-specific, key-free offline workspace for independent humans.

An export is a *private staging artifact*, NOT proof of independent reviewers.
The same Unix UID that owns the original model key must not be permitted to
act as a reviewer. An administrator must provision genuinely different human
accounts, transfer ownership securely, and verify person-level independence.

The returned packet contains only one reviewer's 15 blinded batches plus a
source-content digest manifest. No ground-truth class or model arm mapping,
opposite reviewer sheets, network calls, generated labels, or model weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from src.local_reviewer_terminal import (
    _private_review_directory, _read_private_csv, _read_private_json,
    review_one, _batch_writer_lock,
)
from src.reviewer_batches import ORIGINAL_FIELDS, write_csv
from src.reviewer_handoff_audit import audit as canonical_audit

KIND="lfm3-key-free-human-review-workspace"
MANIFEST="isolated_reviewer_manifest.json"
BATCHES=15
ROWS=90
TOTAL=BATCHES*ROWS
STAGE_ONE_CATEGORIES={"full_compliance","partial_refusal","full_refusal","unclear"}
STAGE_ONE_ACTION={"no","yes","uncertain"}


def _check_private_path(path, directory=False):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("symlinked workspace entry prohibited")
    meta=path.stat()
    if directory:
        if not stat.S_ISDIR(meta.st_mode):
            raise ValueError("workspace must be a directory")
    elif not stat.S_ISREG(meta.st_mode) or meta.st_nlink!=1:
        raise ValueError("workspace file must be single-link regular file")
    if meta.st_mode&0o077:
        raise PermissionError("workspace must be private, 0700 directories and 0600 files")
    return meta


def _original_fingerprint(rows):
    """SHA of original review ID/prompt/response columns only, unaffected by ratings."""
    payload=json.dumps(
        [[row[k] for k in ORIGINAL_FIELDS] for row in rows],
        ensure_ascii=False,separators=(",",":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(workspace):
    workspace=Path(workspace)
    _check_private_path(workspace,directory=True)
    expected={MANIFEST,*[f"batch_{i:03d}.csv" for i in range(1,BATCHES+1)]}
    if {p.name for p in workspace.iterdir()}!=expected:
        raise ValueError("workspace is incomplete or contains unexpected material")
    f=workspace/MANIFEST
    _check_private_path(f)
    m=_read_private_json(f)
    if not isinstance(m,dict) or set(m)!={
        "kind","version","reviewer_slot","exporter_uid","source_sheet_sha256",
        "batch_rows","batch_source_sha256","model_key_in_workspace",
        "human_identity_verified","independence_verified","label_source",
    }:
        raise ValueError("invalid isolated reviewer manifest schema")
    if (m["kind"]!=KIND or m["version"]!=1 or
        type(m["reviewer_slot"]) is not int or m["reviewer_slot"] not in (1,2) or
        type(m["exporter_uid"]) is not int or m["exporter_uid"]<0 or
        m["model_key_in_workspace"] is not False or
        m["human_identity_verified"] is not False or
        m["independence_verified"] is not False or
        m["label_source"]!="human-only" or
        type(m["batch_rows"]) is not int or m["batch_rows"]!=90 or
        not isinstance(m["source_sheet_sha256"],str) or
        len(m["source_sheet_sha256"])!=64 or
        not isinstance(m["batch_source_sha256"],dict) or
        set(m["batch_source_sha256"])!=expected-{MANIFEST}):
        raise ValueError("isolated reviewer metadata failed pinned contract")
    return m


def _batch(workspace, index, *, manifest=None):
    if type(index) is not int or not 1<=index<=BATCHES:
        raise ValueError("batch must be 1 to 15")
    m=manifest if manifest is not None else _manifest(workspace)
    f=Path(workspace)/f"batch_{index:03d}.csv"
    _check_private_path(f)
    rows=_read_private_csv(f)
    if (len(rows)!=ROWS or
        _original_fingerprint(rows)!=m["batch_source_sha256"][f.name]):
        raise ValueError("blinded source columns or batch size changed")
    return f,rows


def inspect(workspace, *, require_distinct_runtime=False):
    """Always show aggregate status; never print prompts or identity strings."""
    m=_manifest(workspace)
    wstat=_check_private_path(workspace,directory=True)
    owners=set()
    initial=0
    remaining=0
    for i in range(1,BATCHES+1):
        f,rows=_batch(workspace,i,manifest=m)
        fstat=_check_private_path(f)
        if fstat.st_uid!=wstat.st_uid:
            raise ValueError("workspace batch owned by a different principal")
        for r in rows:
            if r["reviewer_id"].strip():
                owners.add(r["reviewer_id"].strip())
            valid=(
                bool(r["reviewer_id"].strip()) and
                r["response_category"] in STAGE_ONE_CATEGORIES and
                r["actionable_harmful_content"] in STAGE_ONE_ACTION and
                r["benign_helpfulness_0_3"]==""
            )
            # Enforce phase-one-only workspace: no benchmark-benign labels yet.
            if r["benign_helpfulness_0_3"]!="":
                raise ValueError("benign helpfulness written before gated second stage")
            initial+=int(valid)
            remaining+=int(not valid)
    if len(owners)>1:
        raise ValueError("multiple reviewer identities in one workspace")
    independent_uid=(wstat.st_uid!=m["exporter_uid"])
    runner_is_reviewer=(wstat.st_uid==os.geteuid())
    if require_distinct_runtime and (not independent_uid or
                                     not runner_is_reviewer or
                                     os.geteuid()==0):
        raise PermissionError("separate non-root human reviewer Unix principal required")
    return {
        "kind":"private human-only reviewer workspace status",
        "reviewer_slot":m["reviewer_slot"],
        "source_batches":BATCHES,
        "reviewed_cases":initial,
        "ungraded_cases":remaining,
        "reviewer_identity_present":len(owners)==1,
        "key_material_present":False,
        "workspace_owner_distinct_from_exporter":independent_uid,
        "workspace_owner_is_current_process":runner_is_reviewer,
        "os_account_access_isolated_from_exporter":bool(independent_uid and runner_is_reviewer and os.geteuid()!=0),
        "two_independent_humans_verified":False,
        "human_rating_provenance_verified":False,
        "model_identity_unblinded":False,
        "release_authorized":False,
    }


def export(packet_dir,batches_dir,reviewer,destination,*,source=None):
    """Create a single-reviewer restricted staging bundle without a model key.

    Export's ownership equals the exporting process. It is NOT a reviewer
    workspace until an administrator safely hands it to a distinct real
    human principal. Never overwrite and never touch original packets.
    """
    if type(reviewer) is not int or reviewer not in (1,2):
        raise ValueError("reviewer must be 1 or 2")
    packet,batches=Path(packet_dir),Path(batches_dir)
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite reviewer workspace")
    _check_private_path(destination.parent,directory=True)
    report=canonical_audit(packet,batches,source=source)
    if (not report["original_source_columns_intact"] or
        not report["reviewer_batch_privacy_permissions_verified"] or
        report["reviewer_1"]["batches_with_human_entries"]!=0 or
        report["reviewer_2"]["batches_with_human_entries"]!=0):
        raise ValueError("export must derive from pristine fully verified blank reviewer source")
    batch_contents={}
    frozen={}
    for i in range(1,BATCHES+1):
        f=batches/f"reviewer_{reviewer}"/f"batch_{i:03d}.csv"
        _check_private_path(f)
        rows=_read_private_csv(f)
        if len(rows)!=ROWS or any(any(row[k] for k in (
                "reviewer_id","response_category","actionable_harmful_content",
                "benign_helpfulness_0_3","uncertainty_notes")) for row in rows):
            raise ValueError("reviewer source contains existing human labels")
        batch_contents[f.name]=rows
        frozen[f.name]=_original_fingerprint(rows)
    m={
        "kind":KIND,"version":1,"reviewer_slot":reviewer,
        "exporter_uid":os.geteuid(),
        "source_sheet_sha256":hashlib.sha256(
            (packet/f"blinded_reviewer_{reviewer}.csv").read_bytes()).hexdigest(),
        "batch_rows":ROWS,"batch_source_sha256":frozen,
        "model_key_in_workspace":False,
        "human_identity_verified":False,"independence_verified":False,
        "label_source":"human-only",
    }
    os.umask(0o077)
    destination.mkdir(mode=0o700)
    for name,rows in batch_contents.items():
        write_csv(destination/name,rows)
    path=destination/MANIFEST
    with path.open("x",encoding="utf-8") as file:
        json.dump(m,file,indent=2,sort_keys=True)
        file.write("\n")
    path.chmod(0o600)
    response=inspect(destination)
    if response["key_material_present"] or response["reviewed_cases"]!=0:
        raise ValueError("unexpected staging content")
    return {
        "staging_bundle_created":True,
        "exported_reviewer_slot":reviewer,
        "private_batches":BATCHES,
        "blinded_source_rows":TOTAL,
        "model_key_exported":False,
        "source_annotations_changed":False,
        "independent_reviewer_access_verified":False,
        "human_work_started":False,
        "release_authorized":False,
    }


def review(workspace, reviewer_id, batch=None):
    """Interactive, non-root, different-UID review: stage-one labels only."""
    if not isinstance(reviewer_id,str) or not reviewer_id.strip() or any(
            ord(c)<32 for c in reviewer_id):
        raise ValueError("invalid human reviewer identity")
    if not (os.isatty(0) and os.isatty(1)):
        raise PermissionError("local human interactive TTY required")
    inspect(workspace,require_distinct_runtime=True)
    manifest=_manifest(workspace)
    indexes=(range(1,BATCHES+1) if batch is None else [batch])
    for b in indexes:
        path,rows=_batch(workspace,b,manifest=manifest)
        for idx,r in enumerate(rows):
            if r["response_category"] and r["actionable_harmful_content"]:
                continue
            if not review_one(path,rows,idx,reviewer_id,collect_helpfulness=False):
                return inspect(workspace)
    return inspect(workspace)



def verify_return(packet_dir, batches_dir, workspace, *, source=None):
    """Inspect one complete, distinct-owner stage-one return; never import."""
    packet,batches=Path(packet_dir),Path(batches_dir)
    custody=canonical_audit(packet,batches,source=source)
    if not (custody["original_source_columns_intact"] and
            custody["reviewer_batch_privacy_permissions_verified"]):
        raise ValueError("canonical source custody failed")
    manifest=_manifest(workspace)
    key=_check_private_path(packet/"private_model_key.json")
    owner=_check_private_path(workspace,directory=True)
    if (manifest["exporter_uid"]!=key.st_uid or
            owner.st_uid in (0,key.st_uid)):
        raise PermissionError("distinct non-root reviewer owner required")
    origin=packet/f"blinded_reviewer_{manifest['reviewer_slot']}.csv"
    _check_private_path(origin)
    if hashlib.sha256(origin.read_bytes()).hexdigest()!=manifest["source_sheet_sha256"]:
        raise ValueError("review source fingerprint mismatch")
    summary=inspect(workspace)
    if (summary["reviewed_cases"]!=TOTAL or summary["ungraded_cases"]!=0
            or not summary["reviewer_identity_present"]):
        raise ValueError("incomplete reviewer stage-one ratings")
    for n in range(1,BATCHES+1):
        source_rows=_read_private_csv(
            batches/f"reviewer_{manifest['reviewer_slot']}"/f"batch_{n:03d}.csv")
        _,returned=_batch(workspace,n,manifest=manifest)
        if len(source_rows)!=len(returned) or any(
            a[k]!=b[k] for a,b in zip(source_rows,returned)
            for k in ORIGINAL_FIELDS
        ):
            raise ValueError("reviewed source ID, prompt or response changed")
    return {
        "reviewer_slot":manifest["reviewer_slot"],
        "complete_human_entered_stage_one_fields":True,
        "reviewed_blinded_rows":TOTAL,
        "distinct_unix_owner_from_model_key":True,
        "frozen_review_source_verified":True,
        "original_files_modified":False,
        "reviewer_real_world_identity_independently_verified":False,
        "second_stage_admission_granted":False,
        "model_release_authorized":False,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("export",help="create private single-reviewer staging artifact; not isolated")
    a.add_argument("--packet-dir",type=Path,required=True)
    a.add_argument("--batches-dir",type=Path,required=True)
    a.add_argument("--reviewer",type=int,choices=(1,2),required=True)
    a.add_argument("--destination",type=Path,required=True)
    b=sub.add_parser("status",help="verify private reviewer stage-one packet and source custody")
    b.add_argument("--workspace",type=Path,required=True)
    c=sub.add_parser("review",help="human TTY under a distinct, non-root Unix account")
    c.add_argument("--workspace",type=Path,required=True)
    c.add_argument("--reviewer-id",required=True)
    c.add_argument("--batch",type=int,choices=range(1,16))
    args=p.parse_args()
    if args.action=="export":
        result=export(args.packet_dir,args.batches_dir,args.reviewer,args.destination)
    elif args.action=="status":
        result=inspect(args.workspace)
    else:
        result=review(args.workspace,args.reviewer_id,args.batch)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

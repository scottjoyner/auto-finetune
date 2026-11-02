"""Key-free third-human BENIGN-ONLY helpfulness handoff after adjudication stage one.

Human 0-3 scores are never automatically generated. A different Unix UID or
adjudicator ID is not evidence of a different real human. No model weights,
source model key, original grader labels, analysis or release are exported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src import verified_adjudication_handoff as third
from src import isolated_reviewer_workspace as isolated
from src import isolated_two_stage_merge as merged
from src.local_reviewer_terminal import _read_private_csv, _read_private_json
from src.reviewer_batches import write_csv
from src.reviewer_helpfulness_pass import review_one_helpfulness
from src.xstest_450_eval import read_verified_dataset

MANIFEST="private_third_benign_manifest.json"
SHEET="third_benign_helpfulness.csv"
FROZEN=("review_id","prompt","model_response","reviewer_id",
        "response_category","actionable_harmful_content","uncertainty_notes")


def _hash_rows(rows):
    return hashlib.sha256(json.dumps(
        [[row[k] for k in FROZEN] for row in rows],
        ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()


def _source_stage_one(phase_one):
    m,rows,meta=third._check_workspace(phase_one)
    human_ids={r["reviewer_id"].strip() for r in rows}
    if (len(human_ids)!=1 or not next(iter(human_ids)) or any(
        r["response_category"] not in isolated.STAGE_ONE_CATEGORIES or
        r["actionable_harmful_content"] not in isolated.STAGE_ONE_ACTION or
        r["benign_helpfulness_0_3"]!=""
        for r in rows)):
        raise ValueError("third human must finish ALL blinded initial judgments first")
    if any(row["response_category"]=="unclear" or row["actionable_harmful_content"]=="uncertain" for row in rows):
        raise ValueError("unresolved third-human judgment must be resolved before phase two")
    if (meta.st_uid in (0,m["exporter_uid"],*m["original_reviewer_uids"])):
        raise PermissionError("third-human stage one not owned by independent Unix principal")
    return m,rows,meta


def _source_snapshot(packet,canonical,merge_dir,escrow,first_a,first_b,phase_one,source):
    status=third.verify(packet,canonical,merge_dir,escrow,first_a,first_b,
                        workspace=phase_one,source=source)
    m,rows,meta=_source_stage_one(phase_one)
    if (status["human_first_stage_adjudications_entered"]!=len(rows) or
        not status["third_human_workspace_distinct_unix_owner"]):
        raise ValueError("third-human initial adjudication incomplete or owner unverified")
    safe,key,cases=merged._eligible(packet,canonical,source)
    selected=[dict(r) for r in rows if r["review_id"] in safe]
    if len({r["review_id"] for r in selected})!=len(selected):
        raise ValueError("duplicate benign adjudication ID")
    return m,rows,meta,selected


def _load(workspace):
    workspace=Path(workspace)
    meta=isolated._check_private_path(workspace,directory=True)
    if {p.name for p in workspace.iterdir()}!={SHEET,MANIFEST}:
        raise ValueError("third-human benign workspace has unexpected files")
    for name in (SHEET,MANIFEST):
        f=isolated._check_private_path(workspace/name)
        if f.st_uid!=meta.st_uid:
            raise ValueError("third-human benign workspace files have wrong owner")
    m=_read_private_json(workspace/MANIFEST)
    fields={"kind","schema","exporter_uid","third_owner_uid","original_reviewer_uids",
            "source_initial_manifest_sha256","source_initial_sheet_sha256",
            "selected_benign_row_ids_sha256","frozen_benign_rows_sha256",
            "rows","model_key_exported","original_grades_exported",
            "real_human_identity_attested","release_authorized"}
    if (not isinstance(m,dict) or set(m)!=fields or
        m["kind"]!="third-human-key-free-benign-helpfulness" or
        type(m["schema"]) is not int or m["schema"]!=1 or
        type(m["exporter_uid"]) is not int or m["exporter_uid"]<0 or
        type(m["third_owner_uid"]) is not int or m["third_owner_uid"]<=0 or
        type(m["rows"]) is not int or not 0<=m["rows"]<=750 or
        not isinstance(m["original_reviewer_uids"],list) or
        len(m["original_reviewer_uids"])!=2 or
        any(type(x) is not int or x<=0 for x in m["original_reviewer_uids"]) or
        len(set(m["original_reviewer_uids"]))!=2 or
        m["third_owner_uid"] in (*m["original_reviewer_uids"],m["exporter_uid"]) or
        any(m[k] is not False for k in ("model_key_exported",
            "original_grades_exported","real_human_identity_attested","release_authorized")) or
        any(not isinstance(m[k],str) or len(m[k])!=64 for k in (
            "source_initial_manifest_sha256","source_initial_sheet_sha256",
            "selected_benign_row_ids_sha256","frozen_benign_rows_sha256"))):
        raise ValueError("third-human benign manifest schema or principal invalid")
    rows=_read_private_csv(workspace/SHEET)
    if len(rows)!=m["rows"] or len({r["review_id"] for r in rows})!=len(rows):
        raise ValueError("third-human benign rows missing or duplicated")
    if (merged.second._hash([r["review_id"] for r in rows])!=m["selected_benign_row_ids_sha256"] or
        _hash_rows(rows)!=m["frozen_benign_rows_sha256"]):
        raise ValueError("third-human benign original source/rating context changed")
    if any(r["benign_helpfulness_0_3"] not in ("","0","1","2","3") for r in rows):
        raise ValueError("invalid human benign rating value")
    if len({r["reviewer_id"] for r in rows})>1:
        raise ValueError("mixed third-human reviewer identity")
    return m,rows,meta


def status(workspace,phase_one,*,for_human=False):
    m,rows,meta=_load(workspace)
    sm,original,owner=_source_stage_one(phase_one)
    if (owner.st_uid!=m["third_owner_uid"] or
        sm["original_reviewer_uids"]!=m["original_reviewer_uids"] or
        third._sha_file(Path(phase_one)/third.SHEET)!=m["source_initial_sheet_sha256"] or
        third._sha_file(Path(phase_one)/third.MANIFEST)!=m["source_initial_manifest_sha256"] or
        meta.st_uid!=owner.st_uid):
        raise ValueError("third-human phase-one source, ownership or reviewer ledger drifted")
    originals={r["review_id"]:r for r in original}
    if len(originals)!=len(original) or any(
        r["review_id"] not in originals or
        any(r[k]!=originals[r["review_id"]][k] for k in FROZEN)
        for r in rows):
        raise ValueError("benign source or third-human initial rating differs from frozen stage one")
    separate=(meta.st_uid!=0 and meta.st_uid!=m["exporter_uid"] and
              meta.st_uid not in m["original_reviewer_uids"])
    if for_human and (not separate or os.geteuid()!=meta.st_uid):
        raise PermissionError("third-human helpfulness requires original third non-root human account")
    entered=sum(r["benign_helpfulness_0_3"] in ("0","1","2","3") for r in rows)
    return {
        "benign_disputed_items":len(rows),
        "human_helpfulness_entered":entered,
        "human_helpfulness_remaining":len(rows)-entered,
        "same_separate_third_unix_owner":separate,
        "source_first_stage_unchanged":True,
        "model_key_exported":False,
        "original_reviewer_grades_exported":False,
        "independent_human_identity_attested":False,
        "adjudication_consensus_complete":False,
        "release_authorized":False,
    }


def prepare(packet,canonical,merge_dir,escrow,first_a,first_b,phase_one,destination,*,source=None):
    packet,canonical,merge_dir,escrow,phase_one=map(
        Path,(packet,canonical,merge_dir,escrow,phase_one))
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("refusing to overwrite benign third-human handoff")
    merged.quarantine._safe_parent(destination,
        (packet,canonical,merge_dir,escrow,phase_one))
    source=list(read_verified_dataset() if source is None else source)
    initial,initial_rows,owner,eligible=_source_snapshot(
        packet,canonical,merge_dir,escrow,first_a,first_b,phase_one,source)
    if any(r["benign_helpfulness_0_3"]!="" for r in eligible):
        raise ValueError("helpfulness already scored before eligibility unlock")
    m={
        "kind":"third-human-key-free-benign-helpfulness","schema":1,
        "exporter_uid":os.geteuid(),"third_owner_uid":owner.st_uid,
        "original_reviewer_uids":initial["original_reviewer_uids"],
        "source_initial_manifest_sha256":third._sha_file(phase_one/third.MANIFEST),
        "source_initial_sheet_sha256":third._sha_file(phase_one/third.SHEET),
        "selected_benign_row_ids_sha256":merged.second._hash([r["review_id"] for r in eligible]),
        "frozen_benign_rows_sha256":_hash_rows(eligible),"rows":len(eligible),
        "model_key_exported":False,"original_grades_exported":False,
        "real_human_identity_attested":False,"release_authorized":False,
    }
    os.umask(0o077)
    destination.mkdir(mode=0o700)
    write_csv(destination/SHEET,eligible)
    with (destination/MANIFEST).open("x",encoding="utf-8") as f:
        json.dump(m,f,indent=2,sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    (destination/MANIFEST).chmod(0o600)
    # The operator-owned 0700 directory is only a staging artifact.
    # The real human's UID must own the securely transferred copy before
    # status/review/return checks can claim correct reviewer ownership.
    _,staged,_=_load(destination)
    if any(r["benign_helpfulness_0_3"]!="" for r in staged):
        raise ValueError("third-human benign scores were pre-populated")
    return {
        "key_free_private_staging_created":True,
        "eligible_benign_disputed_items":len(staged),
        "human_helpfulness_entered":0,
        "independent_third_account_handoff_verified":False,
        "actual_human_identity_attested":False,
        "release_authorized":False,
    }


def review(workspace,phase_one,reviewer_id):
    if not (os.isatty(0) and os.isatty(1)):
        raise PermissionError("human benign review requires interactive local TTY")
    status(workspace,phase_one,for_human=True)
    m,rows,_=_load(workspace)
    if (not isinstance(reviewer_id,str) or not reviewer_id.strip() or
        {r["reviewer_id"] for r in rows}!={reviewer_id}):
        raise ValueError("third-human ID differs from original adjudicator")
    f=Path(workspace)/SHEET
    eligible={r["review_id"] for r in rows}
    for i,row in enumerate(rows):
        if row["benign_helpfulness_0_3"]=="":
            if not review_one_helpfulness(f,rows,i,reviewer_id,eligible):
                break
    return status(workspace,phase_one)


def verify_return(packet,canonical,merge_dir,escrow,first_a,first_b,phase_one,
                  workspace,*,source=None):
    """Read-only custodian verification; never import or assert personhood."""
    source=list(read_verified_dataset() if source is None else source)
    initial,initial_rows,owner,expected=_source_snapshot(
        packet,canonical,merge_dir,escrow,first_a,first_b,phase_one,source)
    m,rows,meta=_load(workspace)
    state=status(workspace,phase_one)
    if (not state["same_separate_third_unix_owner"] or
        state["human_helpfulness_remaining"]!=0 or
        m["original_reviewer_uids"]!=initial["original_reviewer_uids"]):
        raise ValueError("third-human helpfulness return incomplete or owner/ledger invalid")
    if [r["review_id"] for r in rows]!=[r["review_id"] for r in expected]:
        raise ValueError("third-human benign source eligibility was altered")
    if any(any(row[k]!=original[k] for k in FROZEN)
           for row,original in zip(rows,expected)):
        raise ValueError("third-human frozen first-stage rating changed")
    return {
        "verified_benign_disputed_items":len(rows),
        "human_entered_benign_scores":len(rows),
        "frozen_adjudicator_first_stage_checked":True,
        "source_eligibility_verified_with_private_key":True,
        "third_account_distinct_from_original_reviewers":True,
        "real_human_identity_independently_attested":False,
        "final_consensus_or_analysis_approved":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    prepare_p=sub.add_parser("prepare")
    for name in ("packet","canonical","merge-dir","escrow","first-a","first-b",
                 "phase-one","destination"):
        prepare_p.add_argument("--"+name,type=Path,required=True)
    state_p=sub.add_parser("status")
    for name in ("workspace","phase-one"):
        state_p.add_argument("--"+name,type=Path,required=True)
    reviewer=sub.add_parser("review")
    for name in ("workspace","phase-one"):
        reviewer.add_argument("--"+name,type=Path,required=True)
    reviewer.add_argument("--reviewer-id",required=True)
    audit=sub.add_parser("verify-return")
    for name in ("packet","canonical","merge-dir","escrow","first-a","first-b",
                 "phase-one","workspace"):
        audit.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.mode=="prepare":
        result=prepare(args.packet,args.canonical,args.merge_dir,args.escrow,
                       args.first_a,args.first_b,args.phase_one,args.destination)
    elif args.mode=="status":
        result=status(args.workspace,args.phase_one)
    elif args.mode=="review":
        result=review(args.workspace,args.phase_one,args.reviewer_id)
    else:
        result=verify_return(args.packet,args.canonical,args.merge_dir,args.escrow,
                             args.first_a,args.first_b,args.phase_one,args.workspace)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

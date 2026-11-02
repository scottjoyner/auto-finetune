"""Provenance-bound, key-free third-human *stage-one* adjudication handoff.

No adjudicator receives the model mapping, either original reviewer's grades,
or the original reviewer's identity. Human personhood and independence are
not proved by a Unix UID. The adjudicator can rate refusal and actionability;
safe-only helpfulness, consensus, statistics and release remain blocked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src import isolated_two_stage_merge as merged
from src import isolated_reviewer_workspace as isolated
from src.local_reviewer_terminal import (
    _open_private_snapshot, _read_private_csv, _read_private_json, review_one,
)
from src.reviewer_adjudication import sources, dispute
from src.human_review_corrections import read_requests
from src.reviewer_batches import FIELDS, ORIGINAL_FIELDS, write_csv
from src.xstest_450_eval import read_verified_dataset

RECEIPT="private_adjudication_handoff_receipt.json"
MANIFEST="isolated_third_human_manifest.json"
SHEET="blinded_adjudication.csv"
STAGING="third_human_staging"


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    with _open_private_snapshot(path) as f:
        return _sha_bytes(f.read().encode("utf-8"))


def _selection(packet,canonical,merge_dir,source):
    merge_dir=Path(merge_dir)
    merged.verify(packet,canonical,merge_dir,source=source)
    # The existing correction path is for reviewer_1/2 canonical batches.
    # Do not silently lose a request that has not been transported to the
    # independently reviewed merged source.
    pending,_=read_requests(packet,canonical)
    if pending:
        raise ValueError("untransported human correction requests require explicit custody")
    lookup={r["id"]:r for r in source}
    if len(lookup)!=450:
        raise ValueError("missing benchmark source prompts")
    key,rows,reviewers,snapshots,requested=sources(
        packet,merge_dir/merged.INNER,lookup)
    if requested:
        raise ValueError("merged review corrections require an explicit handoff")
    selected=[]
    for a,b in zip(*rows):
        if dispute(a,b,lookup[key[a["review_id"]]["source_id"]]["class"]=="safe"):
            selected.append({
                k:a[k] if k in ORIGINAL_FIELDS else "" for k in FIELDS
            })
    # Return only data required by the custodian. Never serialize the key,
    # model arms, original grader names or their response judgments.
    return selected


def _check_workspace(workspace):
    workspace=Path(workspace)
    owner=isolated._check_private_path(workspace,directory=True)
    if {p.name for p in workspace.iterdir()}!={MANIFEST,SHEET}:
        raise ValueError("third-human workspace contains unexpected files")
    for name in (MANIFEST,SHEET):
        if isolated._check_private_path(workspace/name).st_uid!=owner.st_uid:
            raise ValueError("third-human workspace file owner differs")
    m=_read_private_json(workspace/MANIFEST)
    needed={"kind","schema","exporter_uid","items","source_context_sha256",
            "model_key_copied","benchmark_class_disclosed",
            "human_identity_attested","release_authorized"}
    if (not isinstance(m,dict) or set(m)!=needed or
        m["kind"]!="key-free-third-human-refusal-stage-one" or
        type(m["schema"]) is not int or m["schema"]!=1 or
        type(m["exporter_uid"]) is not int or m["exporter_uid"]<0 or
        type(m["items"]) is not int or m["items"]<0 or m["items"]>1350 or
        any(m[k] is not False for k in (
            "model_key_copied","benchmark_class_disclosed",
            "human_identity_attested","release_authorized")) or
        not isinstance(m["source_context_sha256"],str) or
        len(m["source_context_sha256"])!=64):
        raise ValueError("untrusted third-human manifest fields")
    rows=_read_private_csv(workspace/SHEET)
    if len(rows)!=m["items"]:
        raise ValueError("adjudication sheet item count changed")
    return m,rows,owner


def _source_hash(rows):
    return _sha_bytes(json.dumps([
        [row[k] for k in ORIGINAL_FIELDS] for row in rows
    ],ensure_ascii=False,separators=(",",":")).encode("utf-8"))


def prepare(packet_dir,canonical_batches,merge_dir,destination,*,source=None):
    """New private custodian escrow + key-free staging copy, no human claims."""
    packet,canonical,merge_dir=map(Path,(packet_dir,canonical_batches,merge_dir))
    destination=merged.quarantine._safe_parent(
        destination,(packet,canonical,merge_dir))
    source=list(read_verified_dataset() if source is None else source)
    selected=_selection(packet,canonical,merge_dir,source)
    receipt_source=_sha_file(merge_dir/merged.RECEIPT)
    m={
        "kind":"key-free-third-human-refusal-stage-one","schema":1,
        "exporter_uid":os.geteuid(),
        "items":len(selected),
        "source_context_sha256":_source_hash(selected),
        "model_key_copied":False,"benchmark_class_disclosed":False,
        "human_identity_attested":False,"release_authorized":False,
    }
    os.umask(0o077)
    destination.mkdir(mode=0o700)
    human=destination/STAGING
    human.mkdir(mode=0o700)
    write_csv(human/SHEET,selected)
    path=human/MANIFEST
    with path.open("x",encoding="utf-8") as f:
        json.dump(m,f,sort_keys=True,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    path.chmod(0o600)
    receipt={
        "kind":"private-provenance-bound-third-human-handoff",
        "schema":1,
        "merged_receipt_sha256":receipt_source,
        "stage_one_manifest_sha256":_sha_file(path),
        "selected_source_sha256":_source_hash(selected),
        "selected_review_rows":len(selected),
        "original_reviewers_independent_humans_verified":False,
        "third_human_identity_attested":False,
        "third_human_helpfulness_completed":False,
        "human_consensus_approved":False,
        "release_authorized":False,
    }
    target=destination/RECEIPT
    with target.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,sort_keys=True,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    target.chmod(0o600)
    return verify(packet,canonical,merge_dir,destination,source=source)


def verify(packet_dir,canonical_batches,merge_dir,destination,*,workspace=None,
           source=None):
    """Recompute selection against currently verified sealed two-stage source."""
    packet,canonical,merge_dir,destination=map(
        Path,(packet_dir,canonical_batches,merge_dir,destination))
    workspace=Path(workspace) if workspace is not None else destination/STAGING
    isolated._check_private_path(destination,directory=True)
    if {p.name for p in destination.iterdir()}!={STAGING,RECEIPT}:
        raise ValueError("adjudication escrow has incomplete/extra files")
    isolated._check_private_path(destination/RECEIPT)
    receipt=_read_private_json(destination/RECEIPT)
    fields={"kind","schema","merged_receipt_sha256",
            "stage_one_manifest_sha256","selected_source_sha256",
            "selected_review_rows","original_reviewers_independent_humans_verified",
            "third_human_identity_attested","third_human_helpfulness_completed",
            "human_consensus_approved","release_authorized"}
    if (not isinstance(receipt,dict) or set(receipt)!=fields or
        receipt["kind"]!="private-provenance-bound-third-human-handoff" or
        type(receipt["schema"]) is not int or receipt["schema"]!=1 or
        type(receipt["selected_review_rows"]) is not int or
        any(receipt[k] is not False for k in (
            "original_reviewers_independent_humans_verified",
            "third_human_identity_attested","third_human_helpfulness_completed",
            "human_consensus_approved","release_authorized"))):
        raise ValueError("invalid third-human escrow receipt")
    if _sha_file(merge_dir/merged.RECEIPT)!=receipt["merged_receipt_sha256"]:
        raise ValueError("merged source receipt differs from adjudication handoff")
    staging_m,staging_rows,_=_check_workspace(destination/STAGING)
    m,rows,owner=_check_workspace(workspace)
    if (_sha_file(destination/STAGING/MANIFEST)!=receipt["stage_one_manifest_sha256"] or
        m!=staging_m or m["items"]!=receipt["selected_review_rows"]):
        raise ValueError("adjudicator workspace or manifest differs from source")
    source=list(read_verified_dataset() if source is None else source)
    expected=_selection(packet,canonical,merge_dir,source)
    if (_source_hash(expected)!=receipt["selected_source_sha256"] or
        _source_hash(staging_rows)!=_source_hash(expected) or
        _source_hash(rows)!=_source_hash(expected) or
        m["source_context_sha256"]!=_source_hash(expected)):
        raise ValueError("adjudicator source items or canonical dispute set changed")
    if [r["review_id"] for r in rows]!=[r["review_id"] for r in expected]:
        raise ValueError("adjudicator item order/IDs changed")
    seen=set()
    count=0
    for row in rows:
        if row["benign_helpfulness_0_3"]!="":
            raise ValueError("third-human first-stage cannot disclose benign eligibility")
        human=row["reviewer_id"].strip()
        if human:
            seen.add(human)
        if row["response_category"] or row["actionable_harmful_content"]:
            if (not human or
                row["response_category"] not in isolated.STAGE_ONE_CATEGORIES or
                row["actionable_harmful_content"] not in isolated.STAGE_ONE_ACTION):
                raise ValueError("invalid third-human initial rating")
            count+=1
    if len(seen)>1:
        raise ValueError("third-human workspace changed rater identity")
    # A distinct self-reported name is not proof of personhood, but a reused
    # original reviewer ID is an immediate, objectively detectable failure.
    if seen:
        model_key, originals, original_ids, _, _=sources(
            packet,merge_dir/merged.INNER,{r["id"]:r for r in source})
        if seen.intersection(original_ids):
            raise ValueError("third-human ID duplicates an original reviewer")
    isolated_uid=(owner.st_uid!=m["exporter_uid"] and owner.st_uid!=0)
    return {
        "verified_source_selection":True,
        "source_reviewed_model_prompt_pairs":1350,
        "disputed_or_unresolved_human_items":len(expected),
        "human_first_stage_adjudications_entered":count,
        "third_human_workspace_distinct_unix_owner":isolated_uid,
        "source_model_key_not_exported":True,
        "source_benchmark_class_not_exported":True,
        "original_two_humans_independently_attested":False,
        "third_human_identity_independently_attested":False,
        "human_adjudication_fully_complete":False,
        "third_human_benign_helpfulness_complete":False,
        "human_consensus_verified":False,
        "release_authorized":False,
    }


def review(destination,workspace,adjudicator_id):
    """Human-only initial refusal/actionability input from separate account."""
    if not os.isatty(0) or not os.isatty(1):
        raise PermissionError("genuine interactive third-human TTY required")
    m,rows,owner=_check_workspace(workspace)
    if owner.st_uid==m["exporter_uid"] or owner.st_uid==0 or owner.st_uid!=os.geteuid():
        raise PermissionError("third adjudicator must use separately owned non-root account")
    if (not isinstance(adjudicator_id,str) or not adjudicator_id.strip() or
        any(ord(c)<32 for c in adjudicator_id)):
        raise ValueError("invalid human adjudicator ID")
    ids={r["reviewer_id"].strip() for r in rows if r["reviewer_id"].strip()}
    if ids and ids!={adjudicator_id}:
        raise ValueError("adjudicator workspace already belongs to another human")
    for i,row in enumerate(rows):
        if not row["response_category"] or not row["actionable_harmful_content"]:
            if not review_one(workspace/SHEET,rows,i,adjudicator_id,
                              collect_helpfulness=False):
                break
    return {"human_first_stage_entries":sum(
        bool(r["response_category"] and r["actionable_harmful_content"])
        for r in rows),"adjudicator_identity_independently_verified":False,
        "release_authorized":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("prepare","verify"):
        parser=sub.add_parser(mode)
        for field in ("packet-dir","canonical-batches","merge-dir","destination"):
            parser.add_argument("--"+field,type=Path,required=True)
        if mode=="verify":
            parser.add_argument("--workspace",type=Path)
    review_parser=sub.add_parser("review")
    review_parser.add_argument("--destination",type=Path,required=True)
    review_parser.add_argument("--workspace",type=Path,required=True)
    review_parser.add_argument("--adjudicator-id",required=True)
    args=p.parse_args()
    if args.mode=="prepare":
        result=prepare(args.packet_dir,args.canonical_batches,
                       args.merge_dir,args.destination)
    elif args.mode=="verify":
        result=verify(args.packet_dir,args.canonical_batches,
                      args.merge_dir,args.destination,workspace=args.workspace)
    else:
        result=review(args.destination,args.workspace,args.adjudicator_id)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

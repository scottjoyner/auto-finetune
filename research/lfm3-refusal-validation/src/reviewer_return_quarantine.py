"""Private quarantine for two returned, completed human phase-one reviews.

A quarantine is NOT an independent-human attestation, consensus result,
stage-two helpfulness handoff, or an import into canonical reviewer files.
No model key, ground-truth classes or model generation. Original files are
never modified. This tool is intentionally unusable on the current two
same-UID blank review packets.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat

from src import isolated_reviewer_workspace as isolated
from src.local_reviewer_terminal import _open_private_snapshot, _read_private_csv
from src.reviewer_handoff_audit import audit as audit_custody
from src.reviewer_batches import FIELDS, ORIGINAL_FIELDS, write_csv

RECEIPT="private_quarantine_receipt.json"
BATCHDIR="batches"
ORIGINAL_MANIFEST="private_batch_manifest.json"
REVIEWER_DIRS=("reviewer_1","reviewer_2")


def _sha(path):
    with _open_private_snapshot(path) as stream:
        return hashlib.sha256(stream.read().encode("utf-8")).hexdigest()


def _safe_parent(destination, prohibited):
    dest=Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("refusing to overwrite existing private quarantine")
    parent=dest.parent
    for ancestor in (parent,*parent.parents):
        if ancestor.is_symlink():
            raise ValueError("symlinked destination ancestor rejected")
    isolated._check_private_path(parent,directory=True)
    for restricted in prohibited:
        top=Path(restricted).resolve(strict=True)
        proposed=parent.resolve(strict=True)
        if proposed==top or top in proposed.parents:
            raise ValueError("quarantine cannot be created inside original or reviewer workspaces")
    return dest


def _copy_and_verify(originals, return_root, output_dir):
    owner=isolated._manifest(return_root)
    received=[]
    digests={}
    for i in range(1,isolated.BATCHES+1):
        name=f"batch_{i:03d}.csv"
        original=_read_private_csv(originals/name)
        _,rows=isolated._batch(return_root,i,manifest=owner)
        if len(original)!=len(rows) or any(
            a[k]!=b[k] for a,b in zip(original,rows) for k in ORIGINAL_FIELDS
        ):
            raise ValueError("source prompt/response changed in reviewer return")
        if any(
            not r["reviewer_id"].strip() or
            r["response_category"] not in isolated.STAGE_ONE_CATEGORIES or
            r["actionable_harmful_content"] not in isolated.STAGE_ONE_ACTION or
            r["benign_helpfulness_0_3"]!=""
            for r in rows
        ):
            raise ValueError("incomplete or contaminated human first-stage return")
        target=output_dir/name
        write_csv(target,rows)
        digests[name]=_sha(target)
        received.extend(rows)
    if len(received)!=isolated.TOTAL or len({r["review_id"] for r in received})!=isolated.TOTAL:
        raise ValueError("missing or duplicate review IDs")
    if len({r["reviewer_id"] for r in received})!=1:
        raise ValueError("reviewer assignment contains multiple human IDs")
    return digests


def stage(packet_dir,batches_dir,first,second,destination,*,source=None):
    """Stage human reviews into a NEW owner-only quarantine, not canonical.

    The full source/UID/row validation runs before any destination is made.
    A final private receipt is created LAST. Any interruption leaves an
    incomplete private quarantine that cannot pass verify_quarantine().
    """
    packet,batches=Path(packet_dir),Path(batches_dir)
    first,second=Path(first),Path(second)
    destination=_safe_parent(destination,(packet,batches,first,second))
    pair=isolated.verify_pair_returns(
        packet,batches,first,second,source=source)
    if pair["complete_review_workspaces_source_verified"]!=2:
        raise ValueError("two completed blinded reviews required")
    by_slot={}
    for work in (first,second):
        slot=isolated._manifest(work)["reviewer_slot"]
        if slot in by_slot:
            raise ValueError("duplicate reviewer assignment")
        by_slot[slot]=work
    if set(by_slot)!={1,2}:
        raise ValueError("both original review slots required")
    # No labels are imported into the original frozen packets.
    os.umask(0o077)
    parent_fd=os.open(destination.parent,os.O_RDONLY |
                      getattr(os,"O_DIRECTORY",0) |
                      getattr(os,"O_NOFOLLOW",0))
    try:
        fcntl.flock(parent_fd,fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("quarantine destination exists")
        destination.mkdir(mode=0o700)
    finally:
        fcntl.flock(parent_fd,fcntl.LOCK_UN)
        os.close(parent_fd)
    inner=destination/BATCHDIR
    inner.mkdir(mode=0o700)
    original_manifest=batches/ORIGINAL_MANIFEST
    with _open_private_snapshot(original_manifest) as f:
        manifest_data=f.read()
    # Retain the original batch roster/fingerprints; never import the key.
    target=inner/ORIGINAL_MANIFEST
    with target.open("x",encoding="utf-8") as f:
        f.write(manifest_data)
        f.flush()
        os.fsync(f.fileno())
    target.chmod(0o600)
    copy_hash={}
    for n in (1,2):
        folder=inner/f"reviewer_{n}"
        folder.mkdir(mode=0o700)
        copy_hash[str(n)]=_copy_and_verify(
            batches/f"reviewer_{n}",by_slot[n],folder)
    # Ensure source review files have not changed since the copied snapshot.
    isolated.verify_pair_returns(packet,batches,first,second,source=source)
    for n in (1,2):
        for i in range(1,isolated.BATCHES+1):
            name=f"batch_{i:03d}.csv"
            _,current=isolated._batch(by_slot[n],i)
            local=_read_private_csv(inner/f"reviewer_{n}"/name)
            if current!=local:
                raise ValueError("reviewer source changed during quarantine copy")
    custody=audit_custody(packet,inner,source=source)
    if (not custody["original_source_columns_intact"] or
        custody["reviewer_1"]["graded"]!=0 or
        custody["reviewer_2"]["graded"]!=0):
        # First-stage-only safe cases lack helpfulness; this is NOT complete
        # adjudication input, and must not claim behavioral score readiness.
        raise ValueError("unexpected final human-grade readiness in phase-one quarantine")
    receipt={
        "kind":"private-incomplete-human-provenance-stage-one-quarantine",
        "schema":1,"total_return_rows":2700,
        "source_manifest_sha256":_sha(target),
        "reviewer_batch_sha256":copy_hash,
        "private_model_key_copied":False,
        "phase_two_helpfulness_completed":False,
        "independent_real_humans_attested":False,
        "original_frozen_packets_modified":False,
        "canonical_results_imported":False,
        "adjudication_performed":False,
        "release_authorized":False,
    }
    dest=destination/RECEIPT
    with dest.open("x",encoding="utf-8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    dest.chmod(0o600)
    return verify_quarantine(packet, destination, source=source)


def verify_quarantine(packet_dir, destination, *, source=None):
    """No guessing from filenames; verify every private receipt and digest."""
    root=Path(destination)
    isolated._check_private_path(root,directory=True)
    if {p.name for p in root.iterdir()}!={BATCHDIR,RECEIPT}:
        raise ValueError("incomplete/extra quarantine contents")
    inner=root/BATCHDIR
    isolated._check_private_path(inner,directory=True)
    r=root/RECEIPT
    isolated._check_private_path(r)
    with _open_private_snapshot(r) as f:
        receipt=json.load(f)
    if (not isinstance(receipt,dict) or
        set(receipt)!={"kind","schema","total_return_rows","source_manifest_sha256",
                       "reviewer_batch_sha256","private_model_key_copied",
                       "phase_two_helpfulness_completed",
                       "independent_real_humans_attested","original_frozen_packets_modified",
                       "canonical_results_imported","adjudication_performed",
                       "release_authorized"} or
        receipt["kind"]!="private-incomplete-human-provenance-stage-one-quarantine" or
        receipt["schema"]!=1 or receipt["total_return_rows"]!=2700 or
        any(receipt[f] is not False for f in (
            "private_model_key_copied","phase_two_helpfulness_completed",
            "independent_real_humans_attested","original_frozen_packets_modified",
            "canonical_results_imported","adjudication_performed",
            "release_authorized"))):
        raise ValueError("invalid quarantine release/custody manifest")
    expected={ORIGINAL_MANIFEST,*REVIEWER_DIRS}
    if {p.name for p in inner.iterdir()}!=expected:
        raise ValueError("unexpected source or model key inside quarantine")
    if _sha(inner/ORIGINAL_MANIFEST)!=receipt["source_manifest_sha256"]:
        raise ValueError("frozen source manifest hash drift")
    if not isinstance(receipt["reviewer_batch_sha256"],dict) or \
       set(receipt["reviewer_batch_sha256"])!={"1","2"}:
        raise ValueError("missing reviewer batch fingerprints")
    for reviewer in (1,2):
        parent=inner/f"reviewer_{reviewer}"
        isolated._check_private_path(parent,directory=True)
        expected_files={f"batch_{i:03d}.csv" for i in range(1,16)}
        if {x.name for x in parent.iterdir()}!=expected_files:
            raise ValueError("returned private reviewer batch file set changed")
        checks=receipt["reviewer_batch_sha256"][str(reviewer)]
        if not isinstance(checks,dict) or set(checks)!=expected_files:
            raise ValueError("missing sealed reviewer batch digests")
        for name in expected_files:
            isolated._check_private_path(parent/name)
            if _sha(parent/name)!=checks[name]:
                raise ValueError("human reviewer evidence modified after quarantine")
    custody=audit_custody(packet_dir,inner,source=source)
    if (not custody["original_source_columns_intact"] or
        not custody["reviewer_batch_privacy_permissions_verified"] or
        custody["human_review_completed"]):
        raise ValueError("quarantine was misclassified as completed human consensus")
    return {
        "stage_one_quarantine_sealed":True,
        "blinded_human_rating_rows_preserved":2700,
        "reviewer_batch_count":30,
        "original_source_context_verified":True,
        "model_key_included":False,
        "actual_human_independence_attested":False,
        "stage_two_helpfulness_completed":False,
        "canonical_review_packets_modified":False,
        "third_human_adjudication_complete":False,
        "release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    stage_cmd=sub.add_parser("stage")
    stage_cmd.add_argument("--packet-dir",type=Path,required=True)
    stage_cmd.add_argument("--batches-dir",type=Path,required=True)
    stage_cmd.add_argument("--first",type=Path,required=True)
    stage_cmd.add_argument("--second",type=Path,required=True)
    stage_cmd.add_argument("--output-dir",type=Path,required=True)
    check=sub.add_parser("verify")
    check.add_argument("--packet-dir",type=Path,required=True)
    check.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    if args.action=="stage":
        result=stage(args.packet_dir,args.batches_dir,args.first,args.second,
                     args.output_dir)
    else:
        result=verify_quarantine(args.packet_dir,args.output_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

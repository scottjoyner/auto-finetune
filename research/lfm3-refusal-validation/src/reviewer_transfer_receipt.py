"""Private, read-only two-reviewer staging/transfer checksum custody gate.

This tool NEVER copies files or changes user accounts/ACLs. It produces an
owner-only checksum receipt for the original custodian staging bundles, and
can verify that two later administrator-transferred reviewer workspaces are
byte-for-byte identical BEFORE the humans enter any annotations.

Checksums are not signatures or human identity/independence attestations.
Do not publish private receipts, prompts or source identifiers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import stat

from src import isolated_reviewer_workspace as isolated
from src import reviewer_handoff_audit as canonical
from src import reviewer_kernel_access_probe as kernel
from src import reviewer_transfer_preflight as transfer
from src.xstest_450_eval import read_verified_dataset

KIND="private-blank-reviewer-transfer-checksum-receipt"
VERSION=1
FILES=(isolated.MANIFEST,*[f"batch_{i:03d}.csv" for i in range(1,16)])
FLAG_KEYS=("real_humans_independently_verified",
           "os_cross_access_denials_verified","admin_transfer_approved",
           "model_release_authorized")


def _meta(path, directory=False):
    return kernel._private(Path(path),directory=directory)


def _sha(path):
    """Hash exactly one stable owner-only regular inode without following links."""
    p=Path(path)
    # The opened-file identity must come from real kernel metadata, not
    # the separately checked directory-owner ledger.
    original=kernel._private(p)
    if not hasattr(os,"O_NOFOLLOW"):
        raise RuntimeError("Linux O_NOFOLLOW required")
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
    try:
        opened=os.fstat(fd)
        def identity(s):
            return (s.st_dev,s.st_ino,s.st_uid,stat.S_IMODE(s.st_mode),
                    s.st_size,s.st_nlink)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink!=1
                or identity(original)!=identity(opened)):
            raise ValueError("private transfer file changed before checksum")
        digest=hashlib.sha256()
        with os.fdopen(fd,"rb",closefd=False) as stream:
            for part in iter(lambda:stream.read(65536),b""):
                digest.update(part)
        if identity(original)!=identity(os.fstat(fd)):
            raise ValueError("private transfer file changed during checksum")
        return {"sha256":digest.hexdigest(),"size":opened.st_size}
    finally:
        os.close(fd)


def _files(directory,owner):
    directory=Path(directory)
    info=_meta(directory,directory=True)
    if info.st_uid!=owner:
        raise PermissionError("reviewer workspace owner differs from expected principal")
    if stat.S_IMODE(info.st_mode)!=0o700:
        raise PermissionError("reviewer workspace must allow only owner traversal and editing (0700)")
    actual={p.name for p in directory.iterdir()}
    if actual!=set(FILES):
        raise ValueError("unexpected or missing files in reviewer transfer bundle")
    result={}
    for name in FILES:
        item=directory/name
        value=_meta(item)
        if value.st_uid!=owner:
            raise PermissionError("reviewer bundle has mixed Unix ownership")
        if stat.S_IMODE(value.st_mode)!=0o600:
            raise PermissionError("reviewer source files must be owner readable/writable only (0600)")
        result[name]=_sha(item)
    return result


def _original(packet,batches,stages,source):
    packet,batches=Path(packet),Path(batches)
    report=canonical.audit(packet,batches,source=source)
    if (not report["original_source_columns_intact"] or
        not report["reviewer_batch_privacy_permissions_verified"] or
        any(report[f"reviewer_{r}"]["graded"]!=0 or
            report[f"reviewer_{r}"]["batches_with_human_entries"]!=0
            for r in (1,2))):
        raise ValueError("canonical reviewer packets must remain pristine and ungraded")
    model_key_uid=_meta(packet/"private_model_key.json").st_uid
    if Path(stages[0]).resolve(strict=True)==Path(stages[1]).resolve(strict=True):
        raise ValueError("must provide two distinct original staging bundles")
    sizes=[]
    for slot,workspace in enumerate(stages,1):
        sizes.append(transfer._source_export(
            packet,batches,workspace,slot,model_key_uid,source))
    return model_key_uid,sizes


def snapshot(packet,batches,first_stage,second_stage,*,source=None):
    """No content leaves memory except private file SHA-256/size metadata."""
    dataset=list(read_verified_dataset() if source is None else source)
    stages=(Path(first_stage),Path(second_stage))
    owner,sizes=_original(packet,batches,stages,dataset)
    files=[_files(root,owner) for root in stages]
    # Read original custody again after hashing to detect changed sources.
    _,after_sizes=_original(packet,batches,stages,dataset)
    if sizes!=after_sizes or files!=[_files(root,owner) for root in stages]:
        raise ValueError("original staging drifted during transfer receipt snapshot")
    return {
        "kind":KIND,"version":VERSION,"original_key_custodian_uid":owner,
        "reviewer_1":{"slot":1,"files":files[0]},
        "reviewer_2":{"slot":2,"files":files[1]},
        "original_source_verified_and_blank":True,
        "original_staging_owner_only":True,
        "original_blinded_rows_per_reviewer":1350,
        "source_transfer_bytes_per_reviewer":sizes,
        "real_humans_independently_verified":False,
        "os_cross_access_denials_verified":False,
        "admin_transfer_approved":False,
        "model_release_authorized":False,
    }


def verify_staging(receipt,packet,batches,first_stage,second_stage,*,source=None):
    if (not isinstance(receipt,dict) or
        set(receipt)!={"kind","version","original_key_custodian_uid",
                       "reviewer_1","reviewer_2",
                       "original_source_verified_and_blank",
                       "original_staging_owner_only",
                       "original_blinded_rows_per_reviewer",
                       "source_transfer_bytes_per_reviewer",*FLAG_KEYS} or
        receipt.get("kind")!=KIND or type(receipt.get("version")) is not int or
        receipt["version"]!=VERSION or
        any(receipt[k] is not False for k in FLAG_KEYS)):
        raise ValueError("private handoff receipt schema or authority claims invalid")
    latest=snapshot(packet,batches,first_stage,second_stage,source=source)
    if receipt!=latest:
        raise ValueError("private staging receipt changed or source custody drifted")
    return {
        "both_original_blinded_staging_bundles_verified":True,
        "source_blinded_rows_each":1350,
        "staging_reviewer_1_files":16,"staging_reviewer_2_files":16,
        "new_transfer_performed":False,
        "independent_human_identity_attested":False,
        "model_release_authorized":False,
    }


def verify_transfer(receipt,packet,batches,first_stage,second_stage,
                    first_target,second_target,first_uid,second_uid,*,source=None):
    """Read-only post-transfer proof. Does not authorize beginning human work."""
    verify_staging(receipt,packet,batches,first_stage,second_stage,source=source)
    if first_uid==second_uid:
        raise PermissionError("same Unix account cannot hold two human reviews")
    for uid in (first_uid,second_uid):
        if uid==receipt["original_key_custodian_uid"] or uid==0:
            raise PermissionError("reviewer may not be original model key custodian")
        transfer._account(uid)
    stages=(Path(first_stage),Path(second_stage))
    targets=(Path(first_target),Path(second_target))
    if targets[0].resolve(strict=True)==targets[1].resolve(strict=True):
        raise ValueError("both reviewers point to the same transferred workspace")
    for target in targets:
        parent=_meta(target.parent,directory=True)
        if parent.st_uid not in (first_uid,second_uid):
            raise PermissionError("target parent not owned by a designated reviewer")
        for source_root in (Path(packet),Path(batches),*stages):
            actual=source_root.resolve(strict=True)
            path=target.resolve(strict=True)
            if path==actual or actual in path.parents or path in actual.parents:
                raise ValueError("transferred reviewer path overlaps canonical/staging evidence")
    for slot,(dest,uid) in enumerate(zip(targets,(first_uid,second_uid)),1):
        if _meta(dest.parent,directory=True).st_uid!=uid:
            raise PermissionError("transferred reviewer parent UID mismatch")
        recorded=receipt[f"reviewer_{slot}"]["files"]
        current=_files(dest,uid)
        if current!=recorded:
            raise ValueError("transferred reviewer source bytes do not match original staging")
        m=isolated._manifest(dest)
        if m["exporter_uid"]!=receipt["original_key_custodian_uid"]:
            raise ValueError("transferred export lost original key custodian UID ledger")
        checked=isolated.inspect(dest)
        if checked["reviewed_cases"]!=0 or checked["ungraded_cases"]!=1350:
            raise ValueError("human labels appeared before verified transfer")
    verify_staging(receipt,packet,batches,first_stage,second_stage,source=source)
    return {
        "both_private_reviewer_transfers_byte_identical_to_staging":True,
        "distinct_existing_nonroot_reviewer_unix_owners":True,
        "blinded_unscored_rows_per_reviewer":1350,
        "model_key_transferred":False,
        "actual_human_personhood_or_consent_attested":False,
        "cross_reviewer_kernel_access_denials_verified":False,
        "administrator_approved_transfer":False,
        "reviewer_grading_authorized":False,
        "checkpoint_release_authorized":False,
    }


def write_receipt(path,receipt):
    """One-time private artifact under an existing custodian-owned 0700 parent."""
    dest=Path(path)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("refusing to overwrite private handoff receipt")
    if dest.parent.is_symlink():
        raise ValueError("symlinked private receipt parent")
    parent=_meta(dest.parent,directory=True)
    if parent.st_uid!=os.geteuid() or parent.st_uid!=receipt["original_key_custodian_uid"]:
        raise PermissionError("receipt must stay in original custodian account")
    data=(json.dumps(receipt,sort_keys=True,indent=2)+"\n").encode("utf-8")
    fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb",closefd=False) as stream:
            stream.write(data)
            stream.flush()
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "private_receipt_created":True,
        "receipt_file_private":True,
        "reviewer_staging_source_unchanged":True,
        "human_independence_attested":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("prepare","verify-staging","verify-transfer"):
        cli=sub.add_parser(mode)
        for n in ("packet","batches","first-stage","second-stage"):
            cli.add_argument("--"+n,type=Path,required=True)
        if mode=="prepare":
            cli.add_argument("--receipt-file",type=Path,required=True)
        else:
            cli.add_argument("--receipt-file",type=Path,required=True)
        if mode=="verify-transfer":
            for n in ("first-target","second-target"):
                cli.add_argument("--"+n,type=Path,required=True)
            for n in ("first-uid","second-uid"):
                cli.add_argument("--"+n,type=int,required=True)
    args=p.parse_args()
    if args.mode=="prepare":
        fresh=snapshot(args.packet,args.batches,args.first_stage,args.second_stage)
        result=write_receipt(args.receipt_file,fresh)
    else:
        if args.receipt_file.is_symlink():
            raise ValueError("symlinked private receipt")
        _meta(args.receipt_file)
        receipt=json.loads(args.receipt_file.read_text(encoding="utf-8"))
        if args.mode=="verify-staging":
            result=verify_staging(receipt,args.packet,args.batches,
                                  args.first_stage,args.second_stage)
        else:
            result=verify_transfer(receipt,args.packet,args.batches,
                                   args.first_stage,args.second_stage,
                                   args.first_target,args.second_target,
                                   args.first_uid,args.second_uid)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()

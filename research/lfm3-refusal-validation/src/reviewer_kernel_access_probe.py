"""Kernel-open isolation probe for independent XSTest human accounts.

Never read any private file bytes. Preflight is performed by an authorized
custodian with access to the existing model key and three transferred private
reviewer workspaces. Probe must run under each REAL intended unprivileged OS
account with an administrator observing its execution.

No sudo, new accounts, chown, chmod, ACL changes, writes or auto-grading.
A generated JSON status is NOT signed human identity/provenance evidence.
"""
import argparse
import errno
import json
import os
from pathlib import Path
import stat


def _private(path, *, directory=False):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink():
            raise ValueError("isolation source has symlink ancestor")
    info=path.lstat()
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("workspace must be a directory")
    elif not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
        raise ValueError("evidence must be single-link regular file")
    if info.st_mode & 0o077:
        raise PermissionError("source must be private to owner")
    return info


def preflight(packet, reviewer_one, reviewer_two, adjudicator):
    """Admin checks actual path existence, Unix owners and owner-only modes."""
    packet=Path(packet)
    _private(packet,directory=True)
    key=_private(packet/"private_model_key.json")
    folders=[(Path(reviewer_one),"batch_001.csv"),
             (Path(reviewer_two),"batch_001.csv"),
             (Path(adjudicator),"blinded_adjudication.csv")]
    owners=[]
    for path,sample in folders:
        directory=_private(path,directory=True)
        entry=_private(path/sample)
        if directory.st_uid!=entry.st_uid:
            raise PermissionError("workspace and private sample owners differ")
        owners.append(directory.st_uid)
    if any(uid<=0 or uid==key.st_uid for uid in owners) or len(set(owners))!=3:
        raise PermissionError("three actual distinct non-root reviewer owners not present")
    return {
        "kind":"read-only three-account source ownership preflight",
        "canonical_model_key_exists_as_private_regular_file":True,
        "three_distinct_non_root_reviewer_unix_owners":True,
        "reviewer_sample_file_owner_matches_private_directory":True,
        "all_target_paths_exist_and_have_no_symlinks_or_hardlinks":True,
        "private_file_contents_read":False,
        "kernel_denial_probes_executed":False,
        "administrator_witnessed_all_three_live_probes":False,
        "real_independent_human_identities_attested":False,
        "model_release_authorized":False,
    }


def _require_unprivileged_uid(expected):
    """Check real/effective/saved/fsuid AND Linux capability state."""
    if type(expected) is not int or expected<=0 or not hasattr(os,"getresuid"):
        raise PermissionError("Linux non-root UID required")
    if os.getresuid()!=(expected,expected,expected):
        raise PermissionError("real/effective/saved UIDs differ from reviewer account")
    data=Path("/proc/self/status").read_text(encoding="ascii")
    fields=dict(line.split(":",1) for line in data.splitlines() if ":" in line)
    uids=fields.get("Uid","").split()
    if len(uids)!=4 or any(int(x)!=expected for x in uids):
        raise PermissionError("Linux filesystem UID or other process UIDs mismatch")
    try:
        caps=int(fields.get("CapEff","").strip(),16)
    except ValueError as err:
        raise PermissionError("effective capabilities not available") from err
    if caps!=0:
        raise PermissionError("elevated Linux process capabilities forbidden")
    return True


def _open_without_reading(path, *, own_uid=None):
    """Only open/fstat/close: no os.read and no leaking model prompt or key."""
    if not hasattr(os,"O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW required for kernel access proof")
    flags=os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0)
    fd=os.open(os.fspath(path),flags)
    try:
        if own_uid is not None:
            info=os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1
                or info.st_uid!=own_uid or info.st_mode&0o077):
                raise PermissionError("own sample not private or not owned by reviewer")
    finally:
        os.close(fd)


def _require_open_denied(path):
    try:
        _open_without_reading(path)
    except OSError as err:
        if err.errno in (errno.EACCES,errno.EPERM):
            return True
        raise ValueError("ambiguous file open failure is not proof of denial") from err
    raise PermissionError("other reviewer data or private model key was readable")


def probe(role,expected_uid,own_sample,other_first,other_second,key):
    """Run once under each reviewer login; admin must compare real process logs.

    Admin preflight with these exact paths is a separate mandatory operation.
    The returned JSON is not self-authenticating proof of account/personhood.
    """
    if role not in (1,2,3):
        raise ValueError("role must be 1,2 or 3")
    _require_unprivileged_uid(expected_uid)
    paths=[Path(x).absolute() for x in
           (own_sample,other_first,other_second,key)]
    if len(set(paths))!=4:
        raise ValueError("four source paths must be distinct")
    _open_without_reading(paths[0],own_uid=expected_uid)
    _require_open_denied(paths[1])
    _require_open_denied(paths[2])
    _require_open_denied(paths[3])
    return {
        "kind":"live Linux reviewer kernel file-open denial probe",
        "reviewer_role":role,
        "actual_process_uid":expected_uid,
        "own_private_review_file_opened":True,
        "both_other_reviewer_samples_open_denied":True,
        "model_key_open_denied":True,
        "file_contents_read":False,
        "no_linux_effective_capabilities":True,
        "admin_witnessed_preflight_and_process":False,
        "independent_real_person_attestation":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("preflight")
    for name in ("packet","reviewer-one","reviewer-two","adjudicator"):
        a.add_argument("--"+name,type=Path,required=True)
    a=sub.add_parser("probe")
    a.add_argument("--role",type=int,choices=(1,2,3),required=True)
    a.add_argument("--expected-uid",type=int,required=True)
    for name in ("own-sample","other-first","other-second","key"):
        a.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.action=="preflight":
        result=preflight(args.packet,args.reviewer_one,args.reviewer_two,args.adjudicator)
    else:
        result=probe(args.role,args.expected_uid,args.own_sample,
                     args.other_first,args.other_second,args.key)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

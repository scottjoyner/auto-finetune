"""Read-only kernel-access binding to existing reviewer workspaces.

An administrator snapshots *metadata* before and after real separate-UID
file-open probes. A role probe verifies its actual UID and same machine/
namespace, exact original paths, and exact own opened inode. Forbidden target
inode identity cannot be proved by an unprivileged denied open, so this uses
a privileged postflight re-snapshot. A JSON checksum is NOT a signature;
nothing here establishes human identity, trustworthy witnesses, or release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from src import reviewer_kernel_access_probe as kernel

SCHEMA=1


def _digest(obj):
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":"),ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _path_hash(path):
    path=Path(path)
    if not path.is_absolute():
        raise ValueError("preflight and role probes require exact absolute paths")
    return hashlib.sha256(os.fsencode(os.path.normpath(os.fspath(path)))).hexdigest()


def _machine():
    return {
        "boot_id_hash":hashlib.sha256(Path("/proc/sys/kernel/random/boot_id")
             .read_bytes().strip()).hexdigest(),
        "mount_namespace":os.readlink("/proc/self/ns/mnt"),
        "user_namespace":os.readlink("/proc/self/ns/user"),
    }


def _entry(path,*,directory=False):
    checked=kernel._private(path,directory=directory)
    return {
        "path_sha256":_path_hash(path),
        "st_dev":checked.st_dev,
        "st_ino":checked.st_ino,
        "st_uid":checked.st_uid,
        "st_mode":stat.S_IMODE(checked.st_mode),
        "st_nlink":checked.st_nlink,
    }


def _source_paths(packet,one,two,third=None):
    paths={
        "model_key":Path(packet)/"private_model_key.json",
        "reviewer_1":Path(one)/"batch_001.csv",
        "reviewer_2":Path(two)/"batch_001.csv",
    }
    if third is not None:
        paths["reviewer_3"]=Path(third)/"blinded_adjudication.csv"
    return paths


def snapshot(packet,one,two,third=None):
    """Original key and existing workspace ownership, without any file reads."""
    paths=_source_paths(packet,one,two,third)
    for path in paths.values():
        _path_hash(path)
    kernel.preflight(packet,one,two,third)
    return {
        "schema":SCHEMA,
        "kind":"unattested-reviewer-kernel-source-metadata",
        "machine":_machine(),
        "files":{k:_entry(v) for k,v in sorted(paths.items())},
        "human_identity_attested":False,
        "administrator_witnessed_probes":False,
        "reviewer_private_data_read":False,
        "release_authorized":False,
    }


def preflight(packet,one,two,third=None):
    payload=snapshot(packet,one,two,third)
    return {"preflight":payload,"sha256":_digest(payload)}


def _validated_binding(binding):
    if not isinstance(binding,dict) or set(binding)!={"preflight","sha256"}:
        raise ValueError("kernel preflight binding schema invalid")
    p=binding["preflight"]
    if (not isinstance(p,dict) or set(p)!={
        "schema","kind","machine","files","human_identity_attested",
        "administrator_witnessed_probes","reviewer_private_data_read",
        "release_authorized"} or
        type(p["schema"]) is not int or p["schema"]!=SCHEMA or
        p["kind"]!="unattested-reviewer-kernel-source-metadata" or
        any(p[k] is not False for k in ("human_identity_attested",
            "administrator_witnessed_probes","reviewer_private_data_read",
            "release_authorized")) or
        not isinstance(p["machine"],dict) or
        set(p["machine"])!={"boot_id_hash","mount_namespace","user_namespace"} or
        not isinstance(p["files"],dict) or
        set(p["files"]) not in (
            {"model_key","reviewer_1","reviewer_2"},
            {"model_key","reviewer_1","reviewer_2","reviewer_3"}
        ) or binding["sha256"]!=_digest(p)):
        raise ValueError("invalid or changed unsigned preflight binding")
    for item in p["files"].values():
        keys={"path_sha256","st_dev","st_ino","st_uid","st_mode","st_nlink"}
        if (not isinstance(item,dict) or set(item)!=keys or
            not isinstance(item["path_sha256"],str) or
            len(item["path_sha256"])!=64 or
            any(type(item[k]) is not int or item[k]<0 for k in
                ("st_dev","st_ino","st_uid","st_mode","st_nlink"))):
            raise ValueError("preflight source identity incomplete")
    return p


def _role_paths(role,own,other_first,other_second,key,roster):
    expected={
        1:("reviewer_1",{"reviewer_2"} | ({"reviewer_3"} if "reviewer_3" in roster else set())),
        2:("reviewer_2",{"reviewer_1"} | ({"reviewer_3"} if "reviewer_3" in roster else set())),
        3:("reviewer_3",{"reviewer_1","reviewer_2"}),
    }
    if role not in expected:
        raise ValueError("unknown reviewer role")
    mine,others=expected[role]
    if mine not in roster:
        raise ValueError("third reviewer not present in preflight")
    given={"own":own,"key":key}
    other=[x for x in (other_first,other_second) if x is not None]
    if len(other)!=len(others) or len({_path_hash(x) for x in other})!=len(other):
        raise ValueError("the complete original reviewer roster must be probed")
    if _path_hash(own)!=roster[mine]["path_sha256"] or (
        _path_hash(key)!=roster["model_key"]["path_sha256"]
    ) or {_path_hash(x) for x in other}!={roster[k]["path_sha256"] for k in others}:
        raise ValueError("probe paths do not match approved preflight roster")
    return mine,other


def _own_open_matches(path,record,uid):
    """Opening the OWN file must yield exactly the preflight inode/dev."""
    kernel._require_unprivileged_uid(uid)
    fd=os.open(os.fspath(path),
        os.O_RDONLY|os.O_NOFOLLOW|getattr(os,"O_CLOEXEC",0))
    try:
        s=os.fstat(fd)
        if (not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or
            s.st_uid!=uid or s.st_dev!=record["st_dev"] or
            s.st_ino!=record["st_ino"] or
            stat.S_IMODE(s.st_mode)!=record["st_mode"]):
            raise PermissionError("own opened file differs from preflight metadata")
    finally:
        os.close(fd)


def bound_probe(binding,role,expected_uid,own,other_first,other_second,key):
    """Require real role UID, owner match, same namespaces and exact roster."""
    p=_validated_binding(binding)
    if p["machine"]!=_machine():
        raise ValueError("preflight host or mount/user namespace differs")
    roster=p["files"]
    mine,others=_role_paths(role,own,other_first,other_second,key,roster)
    if expected_uid!=roster[mine]["st_uid"]:
        raise PermissionError("runtime reviewer UID does not match preflight")
    _own_open_matches(own,roster[mine],expected_uid)
    # Re-check own open and genuine per-target kernel denial without reads.
    result=kernel.probe(role,expected_uid,own,other_first,other_second,key)
    return {
        "kind":"UNATTESTED bound kernel reviewer open-denial result",
        "source_preflight_sha256":binding["sha256"],
        "real_process_uid":expected_uid,
        "role":role,
        "actual_opened_own_file_matches_preflight_inode":True,
        "same_boot_and_linux_mount_user_namespaces":True,
        "all_probe_path_hashes_match_preflight":True,
        "kernel_denied_other_reviewers_and_model_key":True,
        "original_reviewer_denial_count":result["other_reviewer_samples_tested"],
        "forbidden_target_inodes_same_during_probe_proved":False,
        "postflight_source_identity_checked":False,
        "administrator_witnessed_process":False,
        "independent_real_human_identity_attested":False,
        "model_release_authorized":False,
    }


def postflight(binding,packet,one,two,third=None):
    """Custodian recomputes metadata, not human or probe authenticity."""
    original=_validated_binding(binding)
    new=snapshot(packet,one,two,third)
    if new!=original:
        raise ValueError("source metadata or host context drifted after reviewer probes")
    return {
        "preflight_postflight_metadata_same":True,
        "preflight_sha256":binding["sha256"],
        "probe_execution_witnessed":False,
        "independent_human_identity_attested":False,
        "forbidden_target_inodes_unchanged_throughout_probe_proved":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("preflight","postflight"):
        cli=sub.add_parser(mode)
        for name in ("packet","reviewer-one","reviewer-two"):
            cli.add_argument("--"+name,type=Path,required=True)
        cli.add_argument("--adjudicator",type=Path)
        if mode=="postflight":
            cli.add_argument("--preflight-file",type=Path,required=True)
    role=sub.add_parser("bound-probe")
    role.add_argument("--preflight-file",type=Path,required=True)
    role.add_argument("--role",type=int,choices=(1,2,3),required=True)
    role.add_argument("--expected-uid",type=int,required=True)
    for name in ("own-sample","other-first","key"):
        role.add_argument("--"+name,type=Path,required=True)
    role.add_argument("--other-second",type=Path)
    args=p.parse_args()
    if args.mode=="preflight":
        result=preflight(args.packet,args.reviewer_one,args.reviewer_two,args.adjudicator)
    elif args.mode=="postflight":
        binding=json.loads(args.preflight_file.read_text(encoding="utf-8"))
        result=postflight(binding,args.packet,args.reviewer_one,
                          args.reviewer_two,args.adjudicator)
    else:
        binding=json.loads(args.preflight_file.read_text(encoding="utf-8"))
        result=bound_probe(binding,args.role,args.expected_uid,args.own_sample,
                           args.other_first,args.other_second,args.key)
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()

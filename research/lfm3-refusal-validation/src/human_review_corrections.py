"""Private append-only requests for independent human review corrections.

An original rating is NEVER changed. A request flags its blinded review ID
for the existing third-human adjudication, including when two original
reviewers happened to agree on an erroneous entry. No model invocation,
automatic grading, unblinding, network, or public prompt/label output.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from src.local_reviewer_terminal import (
    _private_review_directory, _open_private_snapshot, open_batch,
)
from src.reviewer_handoff_audit import audit as audit_custody

DIRECTORY = "private_human_correction_requests"
REASONS = ("refusal", "actionability", "helpfulness", "uncertainty")
REVIEW_ID = re.compile(r"XR[0-9]{5}\Z")
NAME = re.compile(r"reviewer_([12])_(XR[0-9]{5})\.json\Z")


def _digest_row(row):
    # The hash attests original evidence without copying prompt or response.
    data = json.dumps(row, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _existing_private_regular(path):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("correction request cannot be a symlink")
    info=path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
        raise ValueError("correction request must be a single-link regular file")
    if info.st_mode & 0o077:
        raise PermissionError("correction request must be owner-only")


def _request_dir(packet_dir, *, create=False):
    base=Path(packet_dir)
    _private_review_directory(base)
    path=base/DIRECTORY
    if create and not path.exists() and not path.is_symlink():
        path.mkdir(mode=0o700)
    if path.exists() or path.is_symlink():
        _private_review_directory(path)
        return path
    return None


def read_requests(packet_dir, batches_dir):
    """Recheck every request and return only in-process blinded IDs + hashes.

    This is a required admission gate for preparing/finalizing adjudication.
    Unexpected objects, modified grades, or malformed requests fail closed.
    """
    root=_request_dir(packet_dir)
    if root is None:
        return set(), {}
    requested=set()
    fingerprints={}
    for path in sorted(root.iterdir(),key=lambda p:p.name):
        m=NAME.fullmatch(path.name)
        if m is None:
            raise ValueError("unexpected human-correction request entry")
        reviewer=int(m.group(1))
        rid=m.group(2)
        _existing_private_regular(path)
        # Read the *verified open inode*, not a path that another writer
        # could replace between stat and read; this stays private.
        try:
            with _open_private_snapshot(path) as stream:
                raw=stream.read().encode("utf-8")
            data=json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("malformed private correction request") from exc
        if not isinstance(data,dict) or set(data)!={
            "schema","reviewer","review_id","batch","index","reason",
            "source_row_sha256","original_rating_unchanged","model_unblinded",
        }:
            raise ValueError("private correction request schema changed")
        if (data["schema"]!=1 or data["reviewer"]!=reviewer
                or data["review_id"]!=rid or data["reason"] not in REASONS
                or data["original_rating_unchanged"] is not True
                or data["model_unblinded"] is not False
                or type(data["batch"]) is not int or not 1<=data["batch"]<=15
                or type(data["index"]) is not int or not 1<=data["index"]<=90):
            raise ValueError("correction request identity or range invalid")
        source_file,rows=open_batch(batches_dir,reviewer,data["batch"],
                                    packet_dir=packet_dir)
        row=rows[data["index"]-1]
        if (row["review_id"]!=rid or
                not row["reviewer_id"].strip() or
                _digest_row(row)!=data["source_row_sha256"]):
            raise ValueError("correction evidence no longer matches original human rating")
        requested.add(rid)
        fingerprints[str(path)]=hashlib.sha256(raw).hexdigest()
    return requested, fingerprints


def request(packet_dir, batches_dir, reviewer, batch, index, reviewer_id, reason, *, source=None):
    """Record human-initiated correction request without touching prior grades."""
    if type(reviewer) is not int or reviewer not in (1,2) or \
       type(batch) is not int or not 1<=batch<=15 or \
       type(index) is not int or not 1<=index<=90 or \
       reason not in REASONS or not isinstance(reviewer_id,str) or \
       not reviewer_id.strip() or any(ord(x)<32 for x in reviewer_id):
        raise ValueError("invalid human correction request")
    # Prior custody check detects source/response/key manipulation and
    # inconsistent reviewer assignments before introducing a request.
    audit_custody(packet_dir,batches_dir,source=source)
    _,rows=open_batch(batches_dir,reviewer,batch,packet_dir=packet_dir)
    row=rows[index-1]
    rid=row["review_id"]
    if not REVIEW_ID.fullmatch(rid):
        raise ValueError("malformed blinded ID")
    if row["reviewer_id"]!=reviewer_id or not row["response_category"] or \
       not row["actionable_harmful_content"]:
        raise ValueError("only the original human can flag a completed annotation")
    root=_request_dir(packet_dir,create=True)
    filename=f"reviewer_{reviewer}_{rid}.json"
    obj={
        "schema":1,"reviewer":reviewer,"review_id":rid,
        "batch":batch,"index":index,"reason":reason,
        "source_row_sha256":_digest_row(row),
        "original_rating_unchanged":True,"model_unblinded":False,
    }
    raw=(json.dumps(obj,sort_keys=True,indent=2)+"\n").encode("utf-8")
    dirfd=os.open(root,os.O_RDONLY | getattr(os,"O_DIRECTORY",0) |
                  getattr(os,"O_NOFOLLOW",0))
    try:
        fcntl.flock(dirfd,fcntl.LOCK_EX)
        flags=(os.O_WRONLY|os.O_CREAT|os.O_EXCL|
               getattr(os,"O_NOFOLLOW",0))
        fd=os.open(filename,flags,0o600,dir_fd=dirfd)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
        finally:
            pass
        os.fsync(dirfd)
    finally:
        fcntl.flock(dirfd,fcntl.LOCK_UN)
        os.close(dirfd)
    return {
        "human_correction_request_created":True,
        "original_human_rating_modified":False,
        "human_adjudicator_required":True,
        "model_identity_unblinded":False,
        "release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir",type=Path,required=True)
    p.add_argument("--batches-dir",type=Path,required=True)
    p.add_argument("--reviewer",type=int,choices=(1,2),required=True)
    p.add_argument("--batch",type=int,choices=range(1,16),required=True)
    p.add_argument("--index",type=int,choices=range(1,91),required=True)
    p.add_argument("--reviewer-id",required=True)
    p.add_argument("--reason",choices=REASONS,required=True)
    args=p.parse_args()
    if not os.isatty(0) or not os.isatty(1):
        p.error("human correction requests require a local interactive TTY")
    result=request(args.packet_dir,args.batches_dir,args.reviewer,args.batch,
                   args.index,args.reviewer_id,args.reason)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

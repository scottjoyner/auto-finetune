"""Read-only OS-principal isolation audit for human XSTest review.

Two separate strings in reviewer_id do not prove two different people, and
0700 directories under the same Unix UID cannot protect reviewers from one
another or from the private model-unblinding key. Never display private
prompts, human identities, source IDs, file contents, or owner usernames.
"""
import argparse
import json
import os
from pathlib import Path
import stat

from src.reviewer_handoff_audit import audit as audit_custody


def _private_stat(path, *, directory):
    p=Path(path)
    if p.is_symlink():
        raise ValueError("symlink in principal isolation audit")
    st=p.stat()
    if directory:
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError("review path is not a directory")
    else:
        if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1:
            raise ValueError("review file must be a single-link regular file")
    if st.st_mode & 0o077:
        raise PermissionError("owner-only review evidence required")
    return st


def audit(packet_dir, batches_dir, *, source=None):
    packet,batches=Path(packet_dir),Path(batches_dir)
    # First prove the source map / 30 reviewer batches are the frozen study.
    custody=audit_custody(packet,batches,source=source)
    key=_private_stat(packet/"private_model_key.json",directory=False)
    srcs=[_private_stat(packet/f"blinded_reviewer_{n}.csv",directory=False)
          for n in (1,2)]
    packet_meta=_private_stat(packet,directory=True)
    batch_meta=_private_stat(batches,directory=True)
    manifest_meta=_private_stat(batches/"private_batch_manifest.json",
                                directory=False)
    dirs=[_private_stat(batches/f"reviewer_{n}",directory=True)
          for n in (1,2)]
    batches_meta=[
        [_private_stat(batches/f"reviewer_{n}"/f"batch_{i:03d}.csv",
                       directory=False) for i in range(1,16)]
        for n in (1,2)
    ]
    # If a reviewer folder and its CSV files have different UIDs the
    # apparent assignment is unusable/ambiguous; never claim isolation.
    per_folder_consistent=[
        all(row.st_uid==d.st_uid for row in rows)
        for d,rows in zip(dirs,batches_meta)
    ]
    reviewers_distinct=dirs[0].st_uid!=dirs[1].st_uid
    key_distinct=all(d.st_uid!=key.st_uid for d in dirs)
    # The current layout has one shared private parent and one private
    # manifest, so independently owned 0700 reviewer subfolders would
    # not be traversable by the two reviewers. A separated staging
    # workflow is required; do not infer readiness merely from chown.
    # Evaluate all claims fail closed. This repository does not yet
    # implement a separate-account verified handoff.
    boundary_verified=False
    return {
        "kind":"read-only Unix principal isolation and human-review custody",
        "source_custody_verified":bool(custody["original_source_columns_intact"] and
                                      custody["reviewer_batch_privacy_permissions_verified"]),
        "reviewers_share_one_unix_uid":not reviewers_distinct,
        "reviewer_folders_distinct_from_model_key_uid":key_distinct,
        "reviewer_directories_match_their_csv_owner":
            all(per_folder_consistent),
        "private_model_key_and_both_reviewer_sources_share_uid":
            key.st_uid==srcs[0].st_uid==srcs[1].st_uid,
        "shared_parent_and_manifest_same_unix_uid_as_key":
            batch_meta.st_uid==manifest_meta.st_uid==key.st_uid,
        "original_packet_private_and_key_owned_by_packet_uid":
            packet_meta.st_uid==key.st_uid,
        "separate_os_principal_handoff_implemented":False,
        "independent_reviewer_access_verified":boundary_verified,
        "physical_independence_of_two_humans_verified":False,
        "review_model_identity_unblinded":False,
        "human_grades_complete":custody["human_review_completed"],
        "release_authorized":False,
        "blockers":[
            "No distinct Unix-UID review workspaces in the current shared 0700 layout",
            "A reviewer ID string does not attest a distinct human reviewer",
            "The study source-model key is accessible to its Unix owner",
            "Separate principal provisioning, custody-preserving handoff, and independent human attestation needed",
        ],
    }


def require_separately_isolated(packet_dir, batches_dir, *, source=None):
    """Never trust a caller-supplied status: re-audit the actual host paths."""
    audit_result = audit(packet_dir, batches_dir, source=source)
    if audit_result["independent_reviewer_access_verified"] is not True or \
       audit_result["physical_independence_of_two_humans_verified"] is not True:
        raise PermissionError(
            "independent human reviewer identities and isolated OS access not established")
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-dir",type=Path,required=True)
    p.add_argument("--batches-dir",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(audit(args.packet_dir,args.batches_dir),indent=2))


if __name__=="__main__":
    main()

"""READ-ONLY preflight for transferring two blank, blinded reviewer exports.

No users/groups are created, no copy/chown/chmod is performed, and no
account/person identity is attested. This admits only already-existing
separate, non-root *local* Unix accounts with private destination parents,
intact key-free 15-batch exports and unchanged canonical benchmark sources.

A PASS only means an authorized administrator can PLAN a transfer. Runtime
cross-account access-denial checks, informed human consent and real-human
identity attestations are still required before actual blinded review.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd

from src import isolated_reviewer_workspace as isolated
from src import reviewer_handoff_audit as canonical
from src import reviewer_kernel_access_probe as kernel
from src.local_reviewer_terminal import _read_private_csv
from src.reviewer_batches import ORIGINAL_FIELDS
from src.xstest_450_eval import read_verified_dataset

EMPTY=("reviewer_id","response_category","actionable_harmful_content",
       "benign_helpfulness_0_3","uncertainty_notes")
EXTRA_BYTES=1024*1024


def _account(uid):
    if type(uid) is not int or uid<=0:
        raise PermissionError("distinct non-root actual local human account UID required")
    try:
        u=pwd.getpwuid(uid)
    except KeyError as exc:
        raise PermissionError("target reviewer account does not exist on this host") from exc
    if u.pw_uid!=uid or u.pw_shell.rsplit("/",1)[-1] in ("nologin","false") or not u.pw_shell:
        raise PermissionError("local reviewer account is not interactive")
    return u


def _no_ancestor_symlink(p):
    p=Path(p).absolute()
    for q in (p,*p.parents):
        if q.is_symlink():
            raise ValueError("symlinked destination or ancestor prohibited")
    return p


def _destination(path,uid,restricted,needed):
    path=_no_ancestor_symlink(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("reviewer destination must be absent")
    parent=path.parent
    owner=isolated._check_private_path(parent,directory=True)
    if owner.st_uid!=uid:
        raise PermissionError("destination parent must already be privately owned by reviewer")
    resolved=parent.resolve(strict=True)
    for original in restricted:
        source=Path(original).resolve(strict=True)
        if resolved==source or source in resolved.parents or resolved in source.parents:
            raise ValueError("destination may not overlap source or another reviewer workspace")
    status=os.statvfs(parent)
    available=status.f_bavail*status.f_frsize
    if available<needed*2+EXTRA_BYTES:
        raise OSError("destination lacks required conservative free-space margin")
    return resolved


def _source_export(packet,canonical_batches,workspace,expected_slot,key_uid,source):
    workspace=Path(workspace)
    m=isolated._manifest(workspace)
    root=isolated._check_private_path(workspace,directory=True)
    if (m["reviewer_slot"]!=expected_slot or m["exporter_uid"]!=key_uid or
        root.st_uid!=key_uid):
        raise PermissionError("source is not a correctly keyed custodian-owned reviewer export")
    origin=Path(packet)/f"blinded_reviewer_{expected_slot}.csv"
    isolated._check_private_path(origin)
    if hashlib.sha256(origin.read_bytes()).hexdigest()!=m["source_sheet_sha256"]:
        raise ValueError("staging packet differs from verified original source sheet")
    observed=isolated.inspect(workspace)
    if (observed["reviewed_cases"]!=0 or observed["ungraded_cases"]!=1350 or
        observed["key_material_present"] or observed["reviewer_identity_present"]):
        raise ValueError("transfer source must be pristine blank and key-free")
    nbytes=0
    for index in range(1,16):
        item,rows=isolated._batch(workspace,index,manifest=m)
        st=isolated._check_private_path(item)
        if st.st_uid!=key_uid:
            raise PermissionError("custodian-owned staging CSV owner differs from manifest")
        canonical_path=Path(canonical_batches)/f"reviewer_{expected_slot}"/item.name
        expected=_read_private_csv(canonical_path)
        if len(expected)!=len(rows) or any(
            any(a[k]!=b[k] for k in ORIGINAL_FIELDS) for a,b in zip(rows,expected)):
            raise ValueError("exported blinded source differs from frozen canonical rows")
        if any(r[field].strip() for r in rows for field in EMPTY):
            raise ValueError("nonblank reviewer rating/identity found in proposed blank transfer")
        nbytes+=st.st_size
    manifest_path=workspace/isolated.MANIFEST
    mstat=isolated._check_private_path(manifest_path)
    if mstat.st_uid!=key_uid:
        raise PermissionError("staging manifest not custodian-owned")
    return nbytes+mstat.st_size


def preflight(packet,canonical_batches,first_stage,second_stage,
              first_destination,second_destination,first_uid,second_uid,
              *,source=None):
    """A readiness check for a FUTURE approved transfer, no mutations."""
    if first_uid==second_uid:
        raise PermissionError("both actual target reviewers cannot share one Unix account")
    _account(first_uid)
    _account(second_uid)
    source=list(read_verified_dataset() if source is None else source)
    reviewed=canonical.audit(packet,canonical_batches,source=source)
    if (not reviewed["original_source_columns_intact"] or
        not reviewed["reviewer_batch_privacy_permissions_verified"] or
        any(reviewed[f"reviewer_{i}"]["graded"] or
            reviewed[f"reviewer_{i}"]["batches_with_human_entries"]
            for i in (1,2))):
        raise ValueError("canonical source must remain fully blank and verified")
    packet=Path(packet)
    key=kernel._private(packet/"private_model_key.json")
    if first_uid in (0,key.st_uid) or second_uid in (0,key.st_uid):
        raise PermissionError("reviewer UID cannot equal custodian/model-key UID")
    if first_stage==second_stage or Path(first_stage).resolve()==Path(second_stage).resolve():
        raise ValueError("source reviewer exports must be two distinct workspaces")
    size_a=_source_export(packet,canonical_batches,first_stage,1,key.st_uid,source)
    size_b=_source_export(packet,canonical_batches,second_stage,2,key.st_uid,source)
    originals=(packet,canonical_batches,first_stage,second_stage)
    a=_destination(first_destination,first_uid,originals,size_a)
    b=_destination(second_destination,second_uid,originals,size_b)
    if a==b or a in b.parents or b in a.parents:
        raise ValueError("target reviewer parents cannot be shared")
    # Snapshot recheck after dest checks; reject modification during preflight.
    if (_source_export(packet,canonical_batches,first_stage,1,key.st_uid,source)!=size_a or
        _source_export(packet,canonical_batches,second_stage,2,key.st_uid,source)!=size_b):
        raise ValueError("staged reviewer exports changed during transfer preflight")
    return {
        "kind":"READ-ONLY proposed dual reviewer transfer preflight",
        "original_study_verified_and_blank":True,
        "both_key_free_staging_packets_source_verified":True,
        "staged_blinded_rows_each":1350,
        "existing_distinct_non_root_local_accounts":True,
        "separate_private_account_owned_destination_parents":True,
        "destination_paths_absent_and_nonoverlapping":True,
        "conservative_destination_space_available":True,
        "source_packets_changed":False,
        "reviewer_workspace_copy_completed":False,
        "reviewer_account_kernel_denials_verified":False,
        "human_identity_and_consent_independently_attested":False,
        "admin_approved_transfer":False,
        "model_release_authorized":False,
    }


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("packet","canonical-batches","first-stage","second-stage",
                "first-destination","second-destination"):
        cli.add_argument("--"+key,type=Path,required=True)
    cli.add_argument("--first-uid",type=int,required=True)
    cli.add_argument("--second-uid",type=int,required=True)
    a=cli.parse_args()
    print(json.dumps(preflight(
        a.packet,a.canonical_batches,a.first_stage,a.second_stage,
        a.first_destination,a.second_destination,a.first_uid,a.second_uid),
        indent=2,sort_keys=True))


if __name__=="__main__":
    main()

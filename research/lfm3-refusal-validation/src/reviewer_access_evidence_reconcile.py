"""Fail-closed two/three-reviewer source-bound access evidence reconciliation.

Only checks that separately generated administrative metadata and reviewer
kernel-open reports agree. JSON may be forged; integrity checks of these
self-reported files are NOT an authenticated administrator witness, proof of
separate real humans, or permission for checkpoint release.

No model-key bytes, prompts, reviewer ratings, OS accounts, or ACLs touched.
"""
import argparse
import json
from pathlib import Path

from src import reviewer_kernel_binding as binding

ROLE_KEYS={
    "kind","source_preflight_sha256","real_process_uid","role",
    "actual_opened_own_file_matches_preflight_inode",
    "same_boot_and_linux_mount_user_namespaces",
    "all_probe_path_hashes_match_preflight",
    "kernel_denied_other_reviewers_and_model_key",
    "original_reviewer_denial_count",
    "forbidden_target_inodes_same_during_probe_proved",
    "postflight_source_identity_checked",
    "administrator_witnessed_process",
    "independent_real_human_identity_attested",
    "model_release_authorized",
}
POST_KEYS={
    "preflight_postflight_metadata_same","preflight_sha256",
    "probe_execution_witnessed","independent_human_identity_attested",
    "forbidden_target_inodes_unchanged_throughout_probe_proved",
    "model_release_authorized",
}
TRUE_ROLE_FIELDS=(
    "actual_opened_own_file_matches_preflight_inode",
    "same_boot_and_linux_mount_user_namespaces",
    "all_probe_path_hashes_match_preflight",
    "kernel_denied_other_reviewers_and_model_key",
)
FALSE_ROLE_FIELDS=(
    "forbidden_target_inodes_same_during_probe_proved",
    "postflight_source_identity_checked",
    "administrator_witnessed_process",
    "independent_real_human_identity_attested",
    "model_release_authorized",
)
FALSE_POST_FIELDS=(
    "probe_execution_witnessed",
    "independent_human_identity_attested",
    "forbidden_target_inodes_unchanged_throughout_probe_proved",
    "model_release_authorized",
)


def _check_role(record,preflight,expected_role,number_of_reviewers):
    """Validate exact source roster, intended UID and strictly limited flags."""
    if not isinstance(record,dict) or set(record)!=ROLE_KEYS:
        raise ValueError("unrecognized role probe fields or omitted evidence")
    files=preflight["preflight"]["files"]
    role=record["role"]
    source_key=f"reviewer_{expected_role}"
    if (type(role) is not int or role!=expected_role or
        type(record["real_process_uid"]) is not int or
        record["real_process_uid"]!=files[source_key]["st_uid"]):
        raise ValueError("role and expected preflight Unix principal mismatch")
    if (record["kind"]!="UNATTESTED bound kernel reviewer open-denial result" or
        record["source_preflight_sha256"]!=preflight["sha256"] or
        any(record[k] is not True for k in TRUE_ROLE_FIELDS) or
        any(record[k] is not False for k in FALSE_ROLE_FIELDS) or
        type(record["original_reviewer_denial_count"]) is not int or
        record["original_reviewer_denial_count"]!=number_of_reviewers-1):
        raise ValueError("role probe evidence is incomplete, forged or claims unproven authority")
    return True


def verify_collected(preflight,roles,postflight):
    """Reconcile unswappable, complete role set; never assert human identity."""
    p=binding._validated_binding(preflight)
    files=p["files"]
    expected={1,2,3} if "reviewer_3" in files else {1,2}
    if not isinstance(roles,list) or len(roles)!=len(expected):
        raise ValueError("missing or duplicate reviewer kernel role evidence")
    by_role={}
    for record in roles:
        if not isinstance(record,dict) or type(record.get("role")) is not int or (
                record["role"] not in expected or record["role"] in by_role):
            raise ValueError("duplicate, unrecognized or invalid role record")
        by_role[record["role"]]=record
    if set(by_role)!=expected:
        raise ValueError("incomplete cross-account role roster")
    for role in expected:
        _check_role(by_role[role],preflight,role,len(expected))
    if not isinstance(postflight,dict) or set(postflight)!=POST_KEYS:
        raise ValueError("postflight receipt is missing or contains unexpected fields")
    if (postflight["preflight_postflight_metadata_same"] is not True or
        postflight["preflight_sha256"]!=preflight["sha256"] or
        any(postflight[k] is not False for k in FALSE_POST_FIELDS)):
        raise ValueError("postflight metadata is changed or claims unauthorized witness evidence")
    # Preflight itself is a mutable JSON digest. A malicious process can
    # rehash fabricated evidence, so no fully verified real OS isolation
    # or human account independence claim can be derived here.
    return {
        "kind":"UNATTESTED offline cross-account access evidence reconciliation",
        "preflight_sha256":preflight["sha256"],
        "complete_distinct_unix_role_count":len(expected),
        "each_role_matches_preflight_uid":True,
        "all_required_bound_kernel_denial_reports_present":True,
        "preflight_postflight_identity_self_report_consistent":True,
        "all_source_file_contents_excluded":True,
        "cryptographic_admin_witness_authority_verified":False,
        "administrator_witnessed_all_actual_processes":False,
        "independent_actual_human_identities_attested":False,
        "live_cross_account_isolation_independently_verified":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight-file",type=Path,required=True)
    p.add_argument("--role-receipt",type=Path,action="append",required=True)
    p.add_argument("--postflight-file",type=Path,required=True)
    a=p.parse_args()
    def load(x):
        return json.loads(Path(x).read_text(encoding="utf-8"))
    print(json.dumps(verify_collected(
        load(a.preflight_file),
        [load(p) for p in a.role_receipt],
        load(a.postflight_file),
    ),indent=2,sort_keys=True))


if __name__=="__main__":
    main()

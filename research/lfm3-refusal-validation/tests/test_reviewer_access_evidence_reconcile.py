"""Synthetic JSON custody tests. NOT trusted human identities or OS probes."""
import copy
import json

import pytest

from src import reviewer_access_evidence_reconcile as reconcile
from src import reviewer_kernel_binding as bound
from tests.test_reviewer_kernel_binding import simulated_accounts


def _report(preflight,role):
    roster=preflight["preflight"]["files"]
    number=3 if "reviewer_3" in roster else 2
    return {
        "kind":"UNATTESTED bound kernel reviewer open-denial result",
        "source_preflight_sha256":preflight["sha256"],
        "real_process_uid":roster[f"reviewer_{role}"]["st_uid"],
        "role":role,
        "actual_opened_own_file_matches_preflight_inode":True,
        "same_boot_and_linux_mount_user_namespaces":True,
        "all_probe_path_hashes_match_preflight":True,
        "kernel_denied_other_reviewers_and_model_key":True,
        "original_reviewer_denial_count":number-1,
        "forbidden_target_inodes_same_during_probe_proved":False,
        "postflight_source_identity_checked":False,
        "administrator_witnessed_process":False,
        "independent_real_human_identity_attested":False,
        "model_release_authorized":False,
    }


def _post(preflight):
    return {
        "preflight_postflight_metadata_same":True,
        "preflight_sha256":preflight["sha256"],
        "probe_execution_witnessed":False,
        "independent_human_identity_attested":False,
        "forbidden_target_inodes_unchanged_throughout_probe_proved":False,
        "model_release_authorized":False,
    }


@pytest.fixture
def scenario(simulated_accounts):
    _,packet,dirs,paths,key=simulated_accounts
    p=bound.preflight(packet,dirs[0],dirs[1])
    return p,[_report(p,1),_report(p,2)],_post(p)


def test_all_two_role_reports_reconciled_but_never_human_attested(scenario):
    p,reports,post=scenario
    result=reconcile.verify_collected(p,reports,post)
    assert result["complete_distinct_unix_role_count"]==2
    assert result["all_required_bound_kernel_denial_reports_present"] is True
    assert result["cryptographic_admin_witness_authority_verified"] is False
    assert result["live_cross_account_isolation_independently_verified"] is False
    assert result["independent_actual_human_identities_attested"] is False
    assert result["model_release_authorized"] is False
    assert "reviewer_id" not in json.dumps(result)


def test_order_irrelevant_but_duplicate_or_missing_role_rejected(scenario):
    p,reports,post=scenario
    assert reconcile.verify_collected(p,reports[::-1],post)["complete_distinct_unix_role_count"]==2
    with pytest.raises(ValueError,match="missing or duplicate"):
        reconcile.verify_collected(p,reports[:1],post)
    with pytest.raises(ValueError,match="duplicate"):
        reconcile.verify_collected(p,[reports[0],reports[0]],post)


@pytest.mark.parametrize("field,value",[
    ("role",3),("real_process_uid",999999),
    ("source_preflight_sha256","tampered"),
    ("actual_opened_own_file_matches_preflight_inode",False),
    ("same_boot_and_linux_mount_user_namespaces",False),
    ("all_probe_path_hashes_match_preflight",False),
    ("kernel_denied_other_reviewers_and_model_key",False),
    ("original_reviewer_denial_count",2),
    ("administrator_witnessed_process",True),
    ("independent_real_human_identity_attested",True),
    ("model_release_authorized",True),
    ("forbidden_target_inodes_same_during_probe_proved",True),
])
def test_role_mismatch_unproven_privileges_and_forged_source_denied(scenario,field,value):
    p,reports,post=scenario
    records=copy.deepcopy(reports)
    records[0][field]=value
    with pytest.raises(ValueError):
        reconcile.verify_collected(p,records,post)


def test_role_extra_data_or_missing_flag_cannot_sneak_by(scenario):
    p,reports,post=scenario
    records=copy.deepcopy(reports)
    records[0]["private_model_key_content"]="not actually a key"
    with pytest.raises(ValueError,match="unrecognized"):
        reconcile.verify_collected(p,records,post)
    del records[0]["private_model_key_content"]
    del records[1]["administrator_witnessed_process"]
    with pytest.raises(ValueError,match="unrecognized"):
        reconcile.verify_collected(p,records,post)


def test_postflight_not_consistent_or_unexpected_human_claim_fails(scenario):
    p,reports,post=scenario
    for field,value in (
        ("preflight_postflight_metadata_same",False),
        ("preflight_sha256","other"),
        ("probe_execution_witnessed",True),
        ("independent_human_identity_attested",True),
        ("forbidden_target_inodes_unchanged_throughout_probe_proved",True),
        ("model_release_authorized",True),
    ):
        altered=dict(post)
        altered[field]=value
        with pytest.raises(ValueError):
            reconcile.verify_collected(p,reports,altered)


def test_missing_postflight_or_unrecognized_schema_fails(scenario):
    p,reports,post=scenario
    with pytest.raises(ValueError):
        reconcile.verify_collected(p,reports,None)
    altered=dict(post)
    altered["independent_permission_proof"]=True
    with pytest.raises(ValueError,match="postflight"):
        reconcile.verify_collected(p,reports,altered)


def test_changed_rehashed_unsigned_preflight_cannot_force_source_approval(scenario):
    p,reports,post=scenario
    changed=copy.deepcopy(p)
    changed["preflight"]["files"]["reviewer_1"]["st_uid"]+=777
    changed["sha256"]=bound._digest(changed["preflight"])
    with pytest.raises(ValueError):
        reconcile.verify_collected(changed,reports,post)
    # Recreating all self-reported receipts could produce another internally
    # consistent package, but that does not establish trusted provenance.
    fabricated=[_report(changed,1),_report(changed,2)]
    result=reconcile.verify_collected(changed,fabricated,_post(changed))
    assert result["live_cross_account_isolation_independently_verified"] is False
    assert result["cryptographic_admin_witness_authority_verified"] is False


def test_three_original_roles_all_required(simulated_accounts):
    _,packet,dirs,paths,key=simulated_accounts
    p=bound.preflight(packet,*dirs)
    docs=[_report(p,i) for i in (1,2,3)]
    result=reconcile.verify_collected(p,docs,_post(p))
    assert result["complete_distinct_unix_role_count"]==3
    assert all(x["original_reviewer_denial_count"]==2 for x in docs)
    with pytest.raises(ValueError,match="missing or duplicate"):
        reconcile.verify_collected(p,docs[:2],_post(p))
    docs[2]["original_reviewer_denial_count"]=1
    with pytest.raises(ValueError,match="incomplete"):
        reconcile.verify_collected(p,docs,_post(p))

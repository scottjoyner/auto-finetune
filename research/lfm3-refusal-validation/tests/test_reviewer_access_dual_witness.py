"""Synthetic two-key Ed25519 access receipt tests; NOT real human witnesses."""
import copy
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from src import reviewer_access_dual_witness as witness
from src import reviewer_access_evidence_reconcile as reconcile
from src import reviewer_kernel_binding as bound
from tests.test_reviewer_access_evidence_reconcile import scenario,_report,_post
from tests.test_reviewer_kernel_binding import simulated_accounts


@pytest.fixture
def signed(scenario):
    preflight,roles,postflight=scenario
    keys=[Ed25519PrivateKey.generate() for _ in range(2)]
    trusted={
        "kind":witness.TRUST_KIND,"schema":1,
        "keys":[{
            "key_id":f"admin-{idx}",
            "public_key_hex":key.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw).hex(),
        } for idx,key in enumerate(keys,1)]
    }
    pin=hashlib.sha256(witness._canonical(trusted)).hexdigest()
    bundle,_=witness._evidence(preflight,roles,postflight)
    message=witness._message(bundle)
    receipt={
        "kind":witness.RECEIPT_KIND,"schema":1,
        "evidence_sha256":hashlib.sha256(witness._canonical(bundle)).hexdigest(),
        "witnesses":[{
            "key_id":f"admin-{idx}",
            "signature_hex":key.sign(message).hex(),
        } for idx,key in enumerate(keys,1)],
        "human_independence_attested":False,
        "administrator_real_world_identity_attested":False,
        "release_authorized":False,
    }
    return preflight,roles,postflight,receipt,trusted,pin,keys


def verify(s):
    p,r,post,receipt,trust,pin,*_=s
    return witness.verify_dual_witness(
        p,r,post,receipt,trust,trusted_roster_sha256=pin)


def test_dual_signed_two_role_receipts_validate_without_human_claim(signed):
    result=verify(signed)
    assert result["complete_required_reviewers"]==2
    assert result["two_distinct_ed25519_public_keys_matched"] is True
    assert result["witness_roster_matches_supplied_out_of_band_digest_pin"] is True
    assert result["witness_operational_process_identity_independently_verified"] is False
    assert result["reviewers_are_distinct_real_humans_attested"] is False
    assert result["reviewer_live_os_access_isolation_independently_verified"] is False
    assert result["model_release_authorized"] is False


def test_role_order_and_signature_order_do_not_change_signed_evidence(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    output=witness.verify_dual_witness(
        p,roles[::-1],post,dict(receipt,witnesses=receipt["witnesses"][::-1]),
        trust,trusted_roster_sha256=pin)
    assert output["two_distinct_ed25519_public_keys_matched"] is True


def test_tampering_role_status_breaks_signed_bundle(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    mutated=copy.deepcopy(roles)
    mutated[0]["role"]=2
    with pytest.raises(ValueError):
        witness.verify_dual_witness(
            p,mutated,post,receipt,trust,trusted_roster_sha256=pin)
    mutated=copy.deepcopy(roles)
    mutated[0]["real_process_uid"]+=1
    with pytest.raises(ValueError):
        witness.verify_dual_witness(
            p,mutated,post,receipt,trust,trusted_roster_sha256=pin)


def test_consistent_self_rehashed_json_still_rejected_by_admin_signature(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    changed=copy.deepcopy(p)
    changed["preflight"]["files"]["reviewer_1"]["st_uid"]+=700
    changed["sha256"]=bound._digest(changed["preflight"])
    changed_roles=[_report(changed,1),_report(changed,2)]
    changed_post=_post(changed)
    # Self-consistent *but invented* technical JSON no longer matches two
    # witness signatures over the original evidence bytes.
    assert reconcile.verify_collected(changed,changed_roles,changed_post)["model_release_authorized"] is False
    changed_bundle,_=witness._evidence(changed,changed_roles,changed_post)
    edited_receipt=dict(receipt,
        evidence_sha256=hashlib.sha256(witness._canonical(changed_bundle)).hexdigest())
    with pytest.raises(PermissionError,match="signature invalid"):
        witness.verify_dual_witness(
            changed,changed_roles,changed_post,edited_receipt,trust,
            trusted_roster_sha256=pin)


@pytest.mark.parametrize("field,value",[
    ("human_independence_attested",True),
    ("administrator_real_world_identity_attested",True),
    ("release_authorized",True),
])
def test_witness_claiming_personhood_or_release_fails_closed(signed,field,value):
    p,r,post,receipt,trust,pin,keys=signed
    altered=copy.deepcopy(receipt)
    altered[field]=value
    with pytest.raises(ValueError,match="overclaiming"):
        witness.verify_dual_witness(p,r,post,altered,trust,trusted_roster_sha256=pin)


def test_missing_or_duplicate_witness_fails_closed(signed):
    p,r,post,receipt,trust,pin,keys=signed
    altered=copy.deepcopy(receipt)
    altered["witnesses"]=altered["witnesses"][:1]
    with pytest.raises(ValueError,match="overclaiming"):
        witness.verify_dual_witness(p,r,post,altered,trust,trusted_roster_sha256=pin)
    altered=copy.deepcopy(receipt)
    altered["witnesses"][1]=dict(altered["witnesses"][0])
    with pytest.raises(ValueError,match="duplicated"):
        witness.verify_dual_witness(p,r,post,altered,trust,trusted_roster_sha256=pin)


def test_unpinned_roster_or_reused_public_key_denied(signed):
    p,r,post,receipt,trust,pin,keys=signed
    altered=copy.deepcopy(trust)
    altered["keys"][1]["public_key_hex"]=altered["keys"][0]["public_key_hex"]
    altered_pin=hashlib.sha256(witness._canonical(altered)).hexdigest()
    with pytest.raises(ValueError,match="same public key"):
        witness.verify_dual_witness(
            p,r,post,receipt,altered,trusted_roster_sha256=altered_pin)
    with pytest.raises(PermissionError,match="roster pin"):
        witness.verify_dual_witness(
            p,r,post,receipt,trust,trusted_roster_sha256="00"*32)


def test_wrong_signature_or_unknown_witness_id_denied(signed):
    p,r,post,receipt,trust,pin,keys=signed
    altered=copy.deepcopy(receipt)
    altered["witnesses"][0]["signature_hex"]="00"*64
    with pytest.raises(PermissionError,match="signature invalid"):
        witness.verify_dual_witness(p,r,post,altered,trust,trusted_roster_sha256=pin)
    altered=copy.deepcopy(receipt)
    altered["witnesses"][0]["key_id"]="unknown-admin"
    with pytest.raises(ValueError,match="pinned roster"):
        witness.verify_dual_witness(p,r,post,altered,trust,trusted_roster_sha256=pin)


def test_three_role_complete_signed_evidence_acceptance_is_not_human_attestation(
        simulated_accounts):
    _,packet,dirs,paths,key=simulated_accounts
    p=bound.preflight(packet,*dirs)
    r=[_report(p,n) for n in (1,2,3)]
    post=_post(p)
    keys=[Ed25519PrivateKey.generate() for _ in range(2)]
    roster={"kind":witness.TRUST_KIND,"schema":1,"keys":[{
        "key_id":f"witness-{i}",
        "public_key_hex":k.public_key().public_bytes(
            serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex(),
    } for i,k in enumerate(keys)]}
    pin=hashlib.sha256(witness._canonical(roster)).hexdigest()
    bundle,_=witness._evidence(p,r,post)
    rec={
        "kind":witness.RECEIPT_KIND,"schema":1,
        "evidence_sha256":hashlib.sha256(witness._canonical(bundle)).hexdigest(),
        "witnesses":[{
            "key_id":f"witness-{i}",
            "signature_hex":k.sign(witness._message(bundle)).hex(),
        } for i,k in enumerate(keys)],
        "human_independence_attested":False,
        "administrator_real_world_identity_attested":False,
        "release_authorized":False,
    }
    result=witness.verify_dual_witness(p,r,post,rec,roster,trusted_roster_sha256=pin)
    assert result["complete_required_reviewers"]==3
    assert result["reviewers_are_distinct_real_humans_attested"] is False


def test_future_evidence_can_never_be_approved_by_internal_signatures_alone(signed):
    result=verify(signed)
    assert result["two_distinct_ed25519_public_keys_matched"] is True
    assert result["witness_administrator_real_world_identity_verified"] is False
    assert result["reviewer_live_os_access_isolation_independently_verified"] is False


def test_offline_request_is_domain_separated_exact_and_contains_no_source_text(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    request=witness.prepare_request(p,roles,post)
    bundle,_=witness._evidence(p,roles,post)
    assert request["domain_separated_ed25519_message_hex"]==witness._message(bundle).hex()
    assert request["evidence_sha256"]==receipt["evidence_sha256"]
    assert request["complete_source_bound_roles"]==2
    assert request["required_different_witness_signatures"]==2
    assert request["administrator_identity_attested"] is False
    assert request["independent_reviewers_are_real_humans_attested"] is False
    assert request["release_authorized"] is False
    assert "reviewer_1" not in str(request)
    assert "model_key" not in str(request)


def test_request_must_validate_all_evidence_before_generating_signing_payload(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    with pytest.raises(ValueError):
        witness.prepare_request(p,roles[:1],post)
    mutated=copy.deepcopy(post)
    mutated["model_release_authorized"]=True
    with pytest.raises(ValueError):
        witness.prepare_request(p,roles,mutated)


def test_request_is_order_invariant_and_signable_only_by_pinned_witnesses(signed):
    p,roles,post,receipt,trust,pin,keys=signed
    first=witness.prepare_request(p,roles,post)
    again=witness.prepare_request(p,roles[::-1],post)
    assert first==again
    opaque=bytes.fromhex(first["domain_separated_ed25519_message_hex"])
    assert keys[0].sign(opaque).hex()==receipt["witnesses"][0]["signature_hex"]

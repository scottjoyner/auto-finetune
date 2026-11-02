"""Ed25519 dual-witness integrity verification for private access audit evidence.

A pre-configured, independently pinned public-key roster is required.
Signatures bind a complete 2/3-role custody evidence bundle to a pair of
different roster keys. The verifier DOES NOT generate keys, sign anything,
execute live OS probes, establish genuine administrator/human identities,
or authorize publication / model release.

Passing signatures means only that private keys corresponding to the
provided pinned roster signed exactly these self-reported bytes. An operator
MUST independently authenticate the trust-root pin and witness identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from src import reviewer_access_evidence_reconcile as audit

DOMAIN=b"LFM3-INDEPENDENT-REVIEWER-ACCESS-WITNESS-v1\x00"
TRUST_KIND="preapproved-out-of-band-access-audit-witness-keys"
RECEIPT_KIND="dual-ed25519-reviewer-access-witness-receipt"
KEY_RE=re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
HEX_RE=re.compile(r"^[0-9a-f]+$")


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode("utf-8")


def _hex(value,size):
    if (not isinstance(value,str) or len(value)!=2*size or
            not HEX_RE.fullmatch(value)):
        raise ValueError("invalid canonical lowercase Ed25519/digest hex")
    return bytes.fromhex(value)


def _keys(trust,expected_roster_sha256):
    if (not isinstance(trust,dict) or set(trust)!={"kind","schema","keys"} or
        trust["kind"]!=TRUST_KIND or
        type(trust["schema"]) is not int or trust["schema"]!=1 or
        not isinstance(trust["keys"],list) or len(trust["keys"])!=2):
        raise ValueError("exactly two independent witness public-key entries required")
    # Pin must come from an independently verified channel. A value generated
    # by the same untrusted person as the roster gives no authority.
    actual=hashlib.sha256(_canonical(trust)).hexdigest()
    if _hex(expected_roster_sha256,32).hex()!=actual:
        raise PermissionError("out-of-band trusted public-key roster pin differs")
    keys={}
    seen_public=set()
    for entry in trust["keys"]:
        if not isinstance(entry,dict) or set(entry)!={"key_id","public_key_hex"}:
            raise ValueError("invalid public witness key entry")
        kid=entry["key_id"]
        if not isinstance(kid,str) or not KEY_RE.fullmatch(kid) or kid in keys:
            raise ValueError("duplicate or invalid trusted witness key identity")
        pub=_hex(entry["public_key_hex"],32)
        if pub in seen_public:
            raise ValueError("two witness identities cannot use the same public key")
        keys[kid]=pub
        seen_public.add(pub)
    return keys


def _evidence(preflight,roles,postflight):
    result=audit.verify_collected(preflight,roles,postflight)
    # Sign exact evidence, not merely a claim that an auditor passed.
    roster=sorted(roles,key=lambda r:r["role"])
    return {
        "kind":"complete-two-or-three-role-access-evidence-v1",
        "preflight":preflight,
        "role_reports":roster,
        "postflight":postflight,
    },result


def _message(evidence):
    return DOMAIN+hashlib.sha256(_canonical(evidence)).digest()


def prepare_request(preflight,roles,postflight):
    """Return an opaque offline signing request, never create private keys."""
    evidence,technical=_evidence(preflight,roles,postflight)
    digest=hashlib.sha256(_canonical(evidence)).hexdigest()
    return {
        "kind":"unattested-private-access-evidence-offline-signing-request",
        "evidence_sha256":digest,
        "domain_separated_ed25519_message_hex":_message(evidence).hex(),
        "required_different_witness_signatures":2,
        "complete_source_bound_roles":technical["complete_distinct_unix_role_count"],
        "administrator_identity_attested":False,
        "independent_reviewers_are_real_humans_attested":False,
        "release_authorized":False,
    }


def verify_dual_witness(preflight,roles,postflight,witness_receipt,
                        trusted_public_keys,*,trusted_roster_sha256):
    """Two independent key signatures; explicitly NOT real-person proof."""
    evidence,technical=_evidence(preflight,roles,postflight)
    pubkeys=_keys(trusted_public_keys,trusted_roster_sha256)
    if (not isinstance(witness_receipt,dict) or set(witness_receipt)!={
        "kind","schema","evidence_sha256","witnesses",
        "human_independence_attested","administrator_real_world_identity_attested",
        "release_authorized"
    } or witness_receipt["kind"]!=RECEIPT_KIND or
        type(witness_receipt["schema"]) is not int or witness_receipt["schema"]!=1 or
        any(witness_receipt[k] is not False for k in (
            "human_independence_attested","administrator_real_world_identity_attested",
            "release_authorized")) or
        not isinstance(witness_receipt["witnesses"],list) or
        len(witness_receipt["witnesses"])!=2):
        raise ValueError("missing or overclaiming signed witness record")
    message=_message(evidence)
    actual=hashlib.sha256(_canonical(evidence)).hexdigest()
    if _hex(witness_receipt["evidence_sha256"],32).hex()!=actual:
        raise ValueError("witness signatures target different evidence bytes")
    # Local import avoids requiring cryptography for unrelated offline
    # research module imports. Hosted research contracts install it directly.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    seen=set()
    for item in witness_receipt["witnesses"]:
        if not isinstance(item,dict) or set(item)!={"key_id","signature_hex"}:
            raise ValueError("unexpected signed witness fields")
        kid=item["key_id"]
        if kid not in pubkeys or kid in seen:
            raise ValueError("witness ID missing from pinned roster or duplicated")
        signature=_hex(item["signature_hex"],64)
        try:
            Ed25519PublicKey.from_public_bytes(pubkeys[kid]).verify(signature,message)
        except (InvalidSignature,ValueError) as exc:
            raise PermissionError("Ed25519 access evidence witness signature invalid") from exc
        seen.add(kid)
    if seen!=set(pubkeys):
        raise ValueError("both independently pinned witness keys required")
    return {
        "kind":"cryptographically dual-signed but not human-attested reviewer access evidence",
        "source_preflight_sha256":preflight["sha256"],
        "complete_required_reviewers":technical["complete_distinct_unix_role_count"],
        "all_bound_role_and_postflight_receipts_consistent":True,
        "two_distinct_ed25519_public_keys_matched":True,
        "witness_roster_matches_supplied_out_of_band_digest_pin":True,
        "evidence_sha256":actual,
        "witness_operational_process_identity_independently_verified":False,
        "witness_administrator_real_world_identity_verified":False,
        "reviewers_are_distinct_real_humans_attested":False,
        "reviewer_live_os_access_isolation_independently_verified":False,
        "model_release_authorized":False,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    for mode in ("request","verify"):
        tool=sub.add_parser(mode)
        for name in ("preflight-file","postflight-file"):
            tool.add_argument("--"+name,type=Path,required=True)
        tool.add_argument("--role-receipt",type=Path,action="append",required=True)
        if mode=="verify":
            tool.add_argument("--witness-receipt-file",type=Path,required=True)
            tool.add_argument("--trusted-public-keys-file",type=Path,required=True)
            tool.add_argument("--trusted-roster-sha256",required=True)
    args=p.parse_args()
    def read(path):
        if path.is_symlink():
            raise ValueError("symlinked witness evidence file")
        return json.loads(path.read_text(encoding="utf-8"))
    preflight=read(args.preflight_file)
    roles=[read(path) for path in args.role_receipt]
    postflight=read(args.postflight_file)
    if args.mode=="request":
        result=prepare_request(preflight,roles,postflight)
    else:
        result=verify_dual_witness(
            preflight,roles,postflight,
            read(args.witness_receipt_file),
            read(args.trusted_public_keys_file),
            trusted_roster_sha256=args.trusted_roster_sha256,
        )
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__=="__main__":
    main()

# Two-witness signed reviewer access evidence — 2026-11-02

**Draft research gate only. No actual independent humans, OS-account probes, witness signatures, real human ratings or checkpoint release have occurred.** This module is solely an integrity verifier for future authorized live-access evidence.

## Evidence chain

First, an authorized custodian performs the existing read-only source-bound preflight, with the original two independently owned reviewer workspaces (optionally a genuine third human). Admin must independently verify the actual original metadata, exact absolute paths, host identity and reviewer accounts.

Second, the custodian obtains an actual source-bound kernel probe report from EACH role under that actual separate non-root Unix login. A probe must check the reviewer-owned sample file, the other reviewer files required by the preflight roster, and the private model-key path; only EACCES/EPERM from Linux counts as denial. Private prompts, model key bytes and review ratings are never read. The custodian reruns source-bound postflight to check that all file metadata is unchanged. The code does NOT automatically authenticate the process operator or prove that denied targets had stable inodes at every instant.

Third, src.reviewer_access_evidence_reconcile verifies ALL self-reported JSON files, not a convenient subset. It rejects omitted/duplicate roles, swapped UIDs, altered preflight hashes, missing denial cases, unexpected fields, false postflight and claims of personhood or release. Its output says UNATTESTED because ordinary JSON checksums can be forged.

Fourth, src.reviewer_access_dual_witness verifies a TWO-SIGNER Ed25519 envelope over the FULL normalized evidence JSON: preflight, exactly two or three individually validated role reports, and matching postflight. Cryptographic verification requires two distinct key IDs, two distinct 32-byte Ed25519 public keys, and an exact 64-hex-character SHA-256 pin of the canonical trusted public-key roster that a user must have authenticated **outside the repository and outside this program**. It is explicitly not enough to accept a key roster or pin supplied by the same untrusted party as the probe receipts. No private signing keys are created, stored, distributed or used by this project.

## Cryptographic format

Canonical JSON encodes UTF-8 with sorted object keys, separators comma/colon, ensure_ascii=false, allow_nan=false. Role reports are sorted by numeric role. The normalized evidence object has kind "complete-two-or-three-role-access-evidence-v1" and keys preflight, role_reports and postflight. Its SHA-256 digest is stored as lowercase 64-character hex.

Each offline witness independently signs this exact byte string using Ed25519:

    ASCII "LFM3-INDEPENDENT-REVIEWER-ACCESS-WITNESS-v1"
    followed by one NUL byte
    followed by the exact 32 raw bytes of SHA256(canonical evidence JSON)

The program can prepare an opaque signing request with this complete domain-separated message encoded in lowercase hex. Only the independent external signing service should hold each private Ed25519 signing key. The repo includes NO private key generation or signing procedure.

Trusted roster JSON (maintained in a separately authorized private system, never accepted as a self-approving proof):

    {
      "kind": "preapproved-out-of-band-access-audit-witness-keys",
      "schema": 1,
      "keys": [
        {"key_id": "approved-witness-one", "public_key_hex": "<64-lowercase-hex>"},
        {"key_id": "approved-witness-two", "public_key_hex": "<64-lowercase-hex>"}
      ]
    }

The trusted-roster SHA-256 PIN is computed over the same canonical representation and independently confirmed before use. Neither the keys nor the operator-supplied pin alone proves the human identity, employment, consent or authority of the witnesses.

Witness receipt JSON contains exact keys kind, schema, evidence_sha256, witnesses, human_independence_attested, administrator_real_world_identity_attested, release_authorized. Kind is "dual-ed25519-reviewer-access-witness-receipt"; schema=1; witnesses is a list of exactly two objects each containing key_id and signature_hex (128-character lowercase hex). The final three booleans MUST be false. An alleged identity or model-release approval in this unsigned/limited scope is rejected outright. The individual record must match an independently pinned trusted public key; duplicate IDs or reused Ed25519 public key bytes are refused.

## Safe operator workflow (after real separate reviewer accounts exist)

From research/lfm3-refusal-validation with the project Python venv and trusted administrator supervision, keep evidence files in separately controlled private directories.

1. Run src.reviewer_kernel_binding preflight with the existing real canonical packet and the independently owned reviewer paths; retain its full private JSON output.
2. Run src.reviewer_kernel_binding bound-probe under each actual designated account, with the full target roster and an approved read-only copy of the preflight, retaining each JSON result.
3. Run src.reviewer_kernel_binding postflight under the authorized custodian and retain its JSON result.
4. Run the following source consistency command with exact paths:

    "$PY" -m src.reviewer_access_evidence_reconcile \
      --preflight-file '<private>/preflight.json' \
      --role-receipt '<private>/role-1.json' \
      --role-receipt '<private>/role-2.json' \
      --postflight-file '<private>/postflight.json'

Add the independently completed role-3 receipt ONLY if the preflight included a genuinely required third reviewer.

5. Prepare an offline signing request — this command does not sign or expose source files:

    "$PY" -m src.reviewer_access_dual_witness request \
      --preflight-file '<private>/preflight.json' \
      --role-receipt '<private>/role-1.json' \
      --role-receipt '<private>/role-2.json' \
      --postflight-file '<private>/postflight.json'

The JSON returned supplies evidence_sha256 and domain_separated_ed25519_message_hex. The independent approved witness operators should verify the underlying admin process logs and source context before signing the exact opaque message using their independent external signer. They must not merely click approve on a request generated by an untrusted operator.

6. Verify an externally returned two-witness receipt and independently approved public-key pin:

    "$PY" -m src.reviewer_access_dual_witness verify \
      --preflight-file '<private>/preflight.json' \
      --role-receipt '<private>/role-1.json' \
      --role-receipt '<private>/role-2.json' \
      --postflight-file '<private>/postflight.json' \
      --witness-receipt-file '<private>/dual-witness.json' \
      --trusted-public-keys-file '<private>/offline-approved-pubkeys.json' \
      --trusted-roster-sha256 '<independently-verified-64-hex-digest>'

No command reads the private model-key content or edits live review packages. The offline Ed25519 library dependency is installed in the repository's research-contract CI; private signing keys are NEVER part of CI.

## Honest security boundary

Correctly verified signatures prove that exactly those public keys correspond to signatures over exactly the evidence bytes. They do NOT, alone, prove the real process role executed what the JSON claims, that different human beings operated accounts, or that the public-key roster was truly established by authorized administrators. The signatures also cannot prove the absence of alternative access routes, a transient target swap restored before postflight, or a separate account with unrestricted network backups.

Even if all these cryptographic tests pass, the result always keeps:
- witness_administrator_real_world_identity_verified = false
- reviewers_are_distinct_real_humans_attested = false
- reviewer_live_os_access_isolation_independently_verified = false
- model_release_authorized = false

Those decisions require separate, trusted operational testimony and independent human/person-level attestations, not model-generated checkmarks.

## Current blocker and test scope

The actual original reviewer batches and model key on x1-370 were confirmed owned by UID 1000. No genuinely separate original reviewer accounts have been provisioned, and live preflight correctly fails. Both original real reviewer assignments still contain 0/1350 human grades. The tests here use synthetic harmless files, mocked UID metadata, ephemeral in-memory Ed25519 test keys and simulated JSON reports. All real source files and model weights remain unchanged.

Draft PR #103 stays DRAFT; P0 source/provenance issue #107 and storage issue #106 remain OPEN. No checkpoint release.

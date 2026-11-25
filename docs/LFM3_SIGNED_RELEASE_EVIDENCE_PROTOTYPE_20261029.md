# LFM3: read-only signed release evidence prototype

Research layer stacked on draft PR #78, after the registry and exporter
release holds. This **does not enable model export, model activation,
adapter publication, training, distribution or any CI bypass**.

## Why

The earlier research train/optimizer receipts are local, synthetic or
self-reported measurements. Even a real, fully loaded LFM3-1.7B BF16
checkpoint smoke test only proves local functionality, not authorization
to publish its weights, independence of held-out evaluation, or a quality
improvement over the baseline.

We need separate proofs of signed authority, source/artifact custody,
non-overlapping held-out evaluation, correct export request binding,
replay prevention, and atomic publication. The prototype implements
only the first *cryptographic syntax / request binding* layer and
explicitly refuses to infer the others.

## Envelope schema

Exact strict JSON fields (no extras, duplicates, NaN or Infinity):

- schema = auto-finetune.release-evidence.v1
- key_id = ASCII lowercase operator-provisioned trust alias
- signature_b64 = strict base64 of a 64-byte Ed25519 signature
- manifest = exactly:
  - release_id: 32 lowercase hex characters; issued_at / expires_at
    as whole-second UTC strings YYYY-MM-DDTHH:MM:SSZ, max 24-hour
    validity, no more than 5-minute future skew
  - base_model_repo and pinned 40-lowercase-hex base_model_revision
  - base_model_sha256, adapter_sha256, training_code_sha256
  - heldout_dataset_sha256 and evaluation_receipt_sha256, each
    exactly 64 lowercase hex characters
  - output_format: merged / gguf / onnx
  - quantization: none for merged/onnx; q8_0, q4_k_m or f16 for gguf
  - output_name: restricted simple basename, never a filesystem path

The Ed25519 signature verifies:
    UTF8("auto-finetune/release-evidence/ed25519/v1\n")
    + ASCII(json.dumps({"schema": SCHEMA, "key_id": kid,
      "manifest": manifest}, sort_keys=True, separators=(",", ":"),
      ensure_ascii=True, allow_nan=False))

JSON keys are unique and are signed with the key_id and schema.
Verification accepts at most 8,192 envelope bytes and rejects any
unknown or improperly typed fields. It checks a separately supplied
exact expected request; a signed adapter approved for GGUF cannot be
reused for ONNX or for an unrelated output name, base revision,
held-out dataset, evaluator receipt or training-code hash.

## Trust and replay limitations

**Trust roots are NOT installed or auto-discovered.** The verifier
requires a mapping of public Ed25519 keys provisioned by a trusted
caller. Untrusted keys embedded in the receipt are not consulted.
No private-key signer, key generator or local key-file loader is
provided in the runtime module. Tests generate ephemeral private
keys only in memory. CI explicitly installs cryptography==43.0.3.

A read-only caller-supplied set of previously used release IDs is
checked and revoked keys are refused, but the function does NOT
atomically claim IDs or persist state. Consequently this is not a
durable replay ledger. Caller-provided timestamps require a trusted
clock (unit tests use a fixed synthetic time). File paths and output
names must also be revalidated by a future secure publisher.

The verifier's successful status is
signature_verified_non_authorizing, with
export_permitted=False and promotion_permitted=False, always.
The ordinary model_export entry points remain blocked (HOLD 78)
regardless of this result. A signature does not make self-reported
metrics independent, does not prove that the hashes correspond to
real artifacts, and does not establish release operator authority.

## Missing production release architecture

1. Separate, independently governed trusted signing authority,
   key rotation, revocation, and authenticated audit log.
2. A verifier that re-hashes model, adapter, train source, evaluator
   and actually sequestered heldout data using safe file-descriptor
   custody; do not accept self-asserted digests as proof.
3. Independent source of evaluation metrics with multi-seed paired
   AdamW/Muon evidence and explicit min-quality/latency guardrails.
4. Transactional, durable, shared replay ledger and time trust,
   checked atomically with one-time release authorization.
5. Isolated temporary export plus deterministic output integrity,
   post-export fidelity tests, protected atomic publication,
   rollback journal, and independently reviewed operator approval.

## Tests and CI

The test suite includes adversarial forged keys, tampering, replay
snapshot checking, missing trust roots, expiry, malformed JSON,
unicode/unknown fields, unsafe output names, request mismatch for
every bound field, no-export assertions, and sensitive data redaction.

Do not lower or override the existing repository-wide 69% coverage
gate. This research module has no access to production services, model
weights, private session transcripts or filesystem-backed keys.

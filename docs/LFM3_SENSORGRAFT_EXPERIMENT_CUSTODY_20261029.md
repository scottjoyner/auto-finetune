# LFM3 / SensorGraft experiment custody — research protocol (2026-10-29)

Research only. This is NOT a SensorGraft ingestion event, a signed production
receipt, a deployed optimizer, or evidence of autonomous model improvement.

## Bound evidence

src/lfm3_sensorgraft_receipt.py creates an explicit UNSIGNED custody manifest
for the real LFM3-1.7B synthetic-token LoRA experiment, after recomputing and
matching its fail-closed review. The manifest binds the exact canonical JSON
SHA-256 hashes of the experiment report and reviewer output, together with
all six optimizer run IDs, each step-telemetry SHA-256, starting and resulting
adapter digests, seed/optimizer IDs and the held-out/train data fingerprints.
It also links the exact official checkpoint revision and SHA-256. An optional
previous-receipt digest can form a future append-only custody chain.

Committed output: research/evidence/lfm3-real-lora-20261029/unsigned-sensorgraft-custody.json.

A hash is NOT a signature; self-reported metrics and provenance are not
independently verified. SensorGraft has NOT ingested this experiment. Flags for
SensorGraft ingestion, semantic/causal verification and weight promotion are
always false.

Create an unsigned manifest without loading a model or GPU:

~~~sh
python3 -m src.lfm3_sensorgraft_receipt \
 research/evidence/lfm3-real-lora-20261029/lfm3-real-lora-3seed-20step-equal-lr-20261029.json \
 research/evidence/lfm3-real-lora-20261029/paired-review.json \
 --output /tmp/unique-lfm3-custody.json
~~~

The output file is new-only; it will never overwrite an existing file.

## Signed Ed25519 receipts

src/lfm3_sensorgraft_verify.py can authenticate a separately supplied
sensorgraft.lfm3.ed25519-receipt.v1 envelope, but ONLY with independently
obtained SHA-256 pins for BOTH the public-key trust-store and revocation
snapshot. The verifier does not trust signer key material embedded in a
producer's receipt. It binds the key ID with the canonical receipt under a
protocol-specific domain separator to prevent relabeling and replay across
other contexts.

External trust-store format:
{ "schema": "sensorgraft.ed25519-trust.v1",
  "keys": { "approved-id": "base64-of-raw-32-byte-public-key" } }

External revocation snapshot:
{ "schema": "sensorgraft.ed25519-revocations.v1",
  "revoked_key_ids": [], "revoked_receipt_sha256": [] }

Both files and their expected hashes must be provided from outside the
experiment repository. Invalid, duplicate, nonfinite, altered, untrusted,
revoked or incomplete inputs fail closed. The verification CLI exits 78
when authentication fails. No production signing key or trusted external
signer has been provisioned for this research artifact. Signature tests use
ephemeral, in-memory Ed25519 keys with no private-key files.

~~~sh
python3 -m src.lfm3_sensorgraft_verify \
 --report /path/to/experiment.json \
 --review /path/to/review.json \
 --signed-envelope /path/to/independent-signature-envelope.json \
 --trust-store /path/to/independent-trust.json \
 --trust-sha256 VERIFIED_TRUST_FILE_SHA256 \
 --revocations /path/to/independent-revocations.json \
 --revocations-sha256 VERIFIED_REVOCATIONS_FILE_SHA256
~~~

Even a valid signature attests only that a trusted public key validates the
exact metadata. It does NOT verify the physical origin, underlying accuracy,
current freshness of revocations, experiment quality or safety, or approve
weight promotion. The verifier returns revocation_freshness_verified=false,
data_origin_independently_verified=false and weight_promotion_permitted=false.

## Native SensorGraft integration gate

Existing Rust trace-memory.v1 is a restricted categorical metadata contract
marked unverified-source-metadata; it rejects neural-weight and causal
claims. Do not write these experiment records into that contract or silently
change its schema.

Next: define a separate versioned Rust receipt importer with independent
signer/trust/revocation checks; store only immutable identity and hash links
in a disposable DB; validate replay, duplicate receipts and revocations;
then independently attest the host/checkpoint/real-data evaluation before
enabling any actual optimizer adjustment.

No live services, model weights, adapter artifacts, private training data,
production credentials, CI admission gates or deployed databases were changed.
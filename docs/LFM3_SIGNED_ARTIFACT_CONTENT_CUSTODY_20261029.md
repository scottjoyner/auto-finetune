# LFM3 signed artifact content custody: read-only POC

Stacks on draft PR #80's non-authorizing signed-evidence and local replay
ledger research. The goal is to check that real bytes of five
caller-selected local artifact files match the hashes carried by a valid
Ed25519 signed proposal. This does not enable model export, promotion,
publication or release approval.

## Inputs and trust boundaries

- Signed envelope bytes, verified by the previously reviewed strict
  Ed25519 research verifier (PR #79).
- An exact, separately supplied expected request and independently
  provisioned public-key mapping. Do not derive either from the
  untrusted signed envelope or an untrusted model metadata file.
- A caller-controlled absolute artifact root in a private 0700
  directory owned by the current process user.
- Exactly five individual digest-field-to-safe-filename mappings:
  base_model_sha256, adapter_sha256, training_code_sha256,
  heldout_dataset_sha256, evaluation_receipt_sha256.

The first absolute path component is limited to /tmp or /home because
x1-370's constrained process policy refuses opening / directly.
It opens the allowlisted initial anchor with O_DIRECTORY and
O_NOFOLLOW and opens every subsequent directory component relative
to a parent descriptor. Files are opened only by basename relative
to the final directory descriptor, with O_NOFOLLOW, O_NONBLOCK,
O_CLOEXEC and read-only flags. A name cannot contain traversal,
path separators, space, Unicode, or doubled dots.

Files must be regular, owned by the process user, singly linked and
no more than 8 GiB individually or 16 GiB combined. All five
descriptors are opened before reading; duplicate filenames and inodes
are rejected. SHA-256 reads at most 1 MiB per chunk, with exact
expected-length enforcement and descriptor metadata checks (inode,
device, link count, owner, mode, size, ctime_ns and mtime_ns) before
and after each file and again when the final file completes.

All results are generic, redacted and immutable. No signing key,
model checkpoint, prompt, evaluation transcript or target path is
written or echoed. Hash mismatches and I/O errors cannot authorize
model release. The existing exporter and model registry still return
HOLD, regardless of hash equality or a local replay claim.

## What a positive result actually means

A valid signature, independently supplied expected request, and
five content digests matched the individual opened local files at
the time of reading. The status is
local_hashes_match_signed_evidence_non_authorizing.

The result still has:
- artifact_source_independently_attested = false
- heldout_independence_verified = false
- fleet_wide_replay_protected = false
- export_permitted = false
- promotion_permitted = false

This is not proof that a signer has release authority, a model is
official or properly licensed, the adapter weights represent the
recorded training run, the heldout dataset was sequestered, or the
evaluation was conducted independently.

## Explicit research POC limitations

1. The prototype verifies **five individual files**. The real
   LFM3 pretrained snapshot consists of multiple source, tokenizer
   and weight files. A complete production bundle manifest must
   authenticate every file, exact path, length and content digest
   (and reject unknown files) rather than using one digest claim.
2. It is not a snapshot/lock of the underlying filesystem.
   A writer or privileged attacker could modify an input before or
   after hashing, or race path replacement. Metadata checks reduce
   mistakes, but cannot defeat all malicious concurrent edits or
   content replacement attacks. Production must use immutable,
   descriptor-custodied object storage and a publisher that operates
   on the *same verified descriptors* rather than reopening paths.
3. The /tmp and /home starting anchors are deployment-specific.
   Other filesystems, mount namespaces, symlinked ancestors or
   process confinement require explicit independent review.
4. Trust roots are injected by the caller. A caller capable of
   injecting its own trusted keys and expected request can fabricate
   synthetic approvals. External governance must bind keys,
   operators, test results, timestamps and revocation.
5. The local replay ledger from PR #80 does not provide distributed,
   fleet-wide one-time release semantics. This file check does not
   claim IDs or write into that ledger.
6. No production data, model file, adapter, export converter, GPU,
   evaluator, network or live registry is exercised. Tests use
   tiny synthetic files plus one sparse oversized-file reject case.

The next architecture should use a complete bundle manifest,
fd-anchored immutable artifact custody and multi-seed heldout
attestation, followed by independent authorization and transactional
atomic publication with rollback.

Do not merge or deploy this draft, relax the 69% coverage gate or
interpret synthetic receipt tests as evidence of model quality.

# LFM3 reviewer handoff receipt and current operational blockers

**2026-11-02 | Draft PR #103 | private custodian-only staging**

## Verified live host status

Read-only enumeration on `x1-370` found exactly one interactive local non-root Unix account, UID 1000. That account also owns the original model key and both reviewer staging exports. The study still has zero human scores in both original 1,350-row reviewer assignments. There is no genuinely separate reviewer account or approved transfer destination on the host; no real human personhood or consent attestation has been gathered.

Two key-free blank staging bundles were created earlier under `/home/scott/git/lfm3-refusal-study-20261029/results/lfm3_reviewer_key_free_staging_20261102`. Each bundle has 15 blind review batches of 90 rows plus a manifest and remains custodian-only.

## New implemented custody boundary

`src.reviewer_transfer_receipt` hashes exactly 16 expected files for each stage-one reviewer using a Linux no-symlink file descriptor. It requires exact owner-only editable 0700 review directories and 0600 review files, one link per file, correct custodian UID, unmodified source fields, no human labels, and full verified canonical benchmark source. The resulting private JSON receipt stores filenames, bytes and SHA-256 digests, not the prompt text or unblinding key. After hashing, it independently rechecks the original source and all file digests. It cannot overwrite an existing receipt.

The receipt was created locally at `/home/scott/git/lfm3-refusal-study-20261029/results/lfm3_reviewer_key_free_staging_20261102/private_transfer_receipt.json`, in the existing 0700 custodian folder, with mode 0600. An immediate read-only `verify-staging` passed on the actual ungraded canonical source and both 16-file staging exports. The receipt is NOT stored in GitHub or a world-readable path.

After a trusted administrator has approved real human reviewers, provisioned two independent non-root Unix accounts and performed a secure private handoff, `verify-transfer` can re-check the receipt and require **byte-for-byte equality of each of the 32 destination files**, matching their original staging source. It also requires separate actual local account UIDs, separate owner-owned private destination parents and 0700/0600 writable privacy modes, correct reviewer slot/manifest/exporter ledger, no key files or early human labels, no overlap with the original source, and another fresh check of the original staged exports.

The post-transfer verifier is READ-ONLY and does not copy, chown, chmod, provision accounts, sign evidence, authenticate administrator identity, verify cross-account access denials, approve a human study, or release model weights. Synthetic tests simulate different Unix owners; they do NOT prove actual independent people.

## Later authorized administrator workflow

From the research directory, only after actual distinct human-operated accounts and correct source-bundle custody exist:

    PY=/home/scott/lfm3-research/train-venv/bin/python
    STUDY=/home/scott/git/lfm3-refusal-study-20261029/results
    STAGE=$STUDY/lfm3_reviewer_key_free_staging_20261102
    "$PY" -m src.reviewer_transfer_receipt verify-staging \
       --packet "$STUDY/xstest_full_review_private" \
       --batches "$STUDY/xstest_review_batched_private" \
       --first-stage "$STAGE/reviewer_1_export" \
       --second-stage "$STAGE/reviewer_2_export" \
       --receipt-file "$STAGE/private_transfer_receipt.json"

An authorized administrator must separately transfer the existing key-free exports to genuine reviewer-owned private destinations. After the authorized transfer and before grading:

    "$PY" -m src.reviewer_transfer_receipt verify-transfer \
       --packet "$STUDY/xstest_full_review_private" \
       --batches "$STUDY/xstest_review_batched_private" \
       --first-stage "$STAGE/reviewer_1_export" \
       --second-stage "$STAGE/reviewer_2_export" \
       --receipt-file "$STAGE/private_transfer_receipt.json" \
       --first-target '<private-human-one-destination>' \
       --second-target '<private-human-two-destination>' \
       --first-uid '<approved-human-one-existing-local-uid>' \
       --second-uid '<approved-human-two-existing-local-uid>'

After successful data equality, separate real-person identity and consent attestation and actual trusted-owner kernel cross-access denial remain required. The two-signature witness code does not automatically prove either fact.

## Explicit nonclaims

Private staged receipts are content digests, not independently trusted signatures. The tool intentionally refuses same-UID and key-owner handoff. No reviewer has begun human grading, no OS user account was created, no original source contents were changed, no model weights were modified, and no release is authorized. P0 issue #107 and storage issue #106 remain open; draft PR #103 is not ready for approval.

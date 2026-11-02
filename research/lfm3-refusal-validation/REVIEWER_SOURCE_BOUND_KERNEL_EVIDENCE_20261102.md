# LFM3 reviewer OS source-bound access evidence — 2026-11-02

**Status: synthetic-path tested only; independent human reviewer accounts NOT provisioned.** No live review data or private model-key content is read or changed.

## Three read-only steps

1. Custodian PRE-FLIGHT: verify real existing owner-only model-key and reviewer sample files, distinct non-root Unix owners, single-link regular files, no symlinks. Record SHA-256 hashes of exact absolute paths, device/inode/UID/mode/link metadata, Linux boot-ID hash, mount namespace and user namespace. The output JSON has a self-checking SHA-256 digest only; it is NOT signed or trusted-person attestation.
2. Reviewer BOUND-PROBE: run under each actually designated non-root human-operated Unix login, on the same host and mount/user namespace. Validate the exact preflight roster digest, current real/effective/saved/filesystem UIDs, no effective capabilities, exact role/target path hashes, and the own-file opened inode. Attempt kernel read-only open of every required other reviewer sample and the private model-key file; only genuine EACCES/EPERM is accepted. No private data is read. Role three demands both original reviewer paths.
3. Custodian POST-FLIGHT: independently re-stat all EXISTING paths and require the same identities/owners/inodes, modes, host and namespace. Negative opens cannot prove inaccessible target inode identity at the exact moment of the denied open; a transient swap/restore remains theoretically possible. Results never claim real human personhood, administrator witnessing, signed evidence or model release.

## Operator use — after real separate human accounts have been approved and provisioned

From the research Python environment, the custodian uses an existing owner-only output directory, not canonical reviewer folders:

    PY=/home/scott/lfm3-research/train-venv/bin/python
    PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
    umask 077
    "$PY" -m src.reviewer_kernel_binding preflight \
      --packet "$PACKET" \
      --reviewer-one '<real-reviewer-1-owned-workspace>' \
      --reviewer-two '<real-reviewer-2-owned-workspace>' \
      > '<existing-private-custodian-dir>/source-binding.json'

The custodian records that unsigned checksum evidence and transfers a safe, read-only copy of the noncontent roster to each reviewer account. Under actual reviewer one, under administrator-supervised execution:

    "$PY" -m src.reviewer_kernel_binding bound-probe \
      --preflight-file '<reviewer-readable-copy-of-source-binding.json>' \
      --role 1 --expected-uid '<actual-role-1-uid>' \
      --own-sample '<reviewer-1-owned-workspace>/batch_001.csv' \
      --other-first '<reviewer-2-owned-workspace>/batch_001.csv' \
      --key "$PACKET/private_model_key.json"

Run role two under its actual Unix account, swapping own/other samples and UID. If human adjudication actually needs a separate third person, redo preflight including --adjudicator pointing at the third human's private first-stage workspace; test both other reviewer samples with --other-first and --other-second for all three roles.

Then the custodian postflight checks the same source roots and original roster:

    "$PY" -m src.reviewer_kernel_binding postflight \
      --preflight-file '<existing-private-custodian-dir>/source-binding.json' \
      --packet "$PACKET" \
      --reviewer-one '<real-reviewer-1-owned-workspace>' \
      --reviewer-two '<real-reviewer-2-owned-workspace>'

Include --adjudicator for a three-account preflight. An administrator must separately witness the actual process account mapping, record signed/trusted evidence and establish actual independent persons. The JSON SHA-256 hash is not a digital signature.

## Live status and limitations

The original canonical source on x1-370 has model-key and both original reviewer samples owned by the SAME UID 1000, so no two-reviewer live preflight or trusted separate-account denial is possible yet. Source-bound unit tests use harmless local synthetic files, simulated distinct UIDs and true Linux denied opens against synthetic inaccessible files. These tests do not prove real three-person independence.

No accounts, permissions, model weights, rating entries or other source content were changed. Keep P0 reviewer isolation #107 OPEN, storage #106 OPEN and draft PR #103 NOT releasable.

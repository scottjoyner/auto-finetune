# LFM3-1.7B: Key-free independent reviewer workspace handoff — 2026-11-01

**Implemented and synthetic-tested, not operationally handed to human reviewers.** Original A/B/C frozen responses, the 30 private canonical batches, and source model key remain untouched. Both original reviewers are still 0/1350 graded.

## Artifact boundaries

`src/isolated_reviewer_workspace.py` exports ONE reviewer slot into a newly created private directory, containing exactly `batch_001.csv` through `batch_015.csv` plus `isolated_reviewer_manifest.json`. Each batch contains 90 blinded prompt/response records with blank human rating fields. The manifest pins the SHA-256 digest of each batch's three immutable original columns and source reviewer sheet; it records exporting Unix UID and explicitly asserts `model_key_in_workspace=false`, `human_identity_verified=false`, `independence_verified=false`. Neither arm identities, safe/unsafe ground truth, the other reviewer's material, nor any model weights appear in the bundle.

An export is a **same-owner private staging copy**, not proof of isolation. It must NOT be treated as independently blinded review until separate actual human accounts are provisioned and secure cross-owner handoff is executed. Do not make a staging copy merely to claim issue #107 is closed.

## Administrator procedure (future, authorized only)

1. An authorized administrator provisions/selects two genuinely separate human-operated Unix accounts, each with private login, and records real-world identity attestation outside GitHub. Existing UID-1000 user must not conduct both reviews.
2. Choose separate staging parents inaccessible to unauthorized users. Export reviewer 1 and reviewer 2 **individually**, checking no existing files are overwritten and no prompt/response data are logged:

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Destination parent MUST already exist with mode 0700 and appropriate ownership.
# Hypothetical approved paths only: do not run before real account provisioning.
"$PY" -m src.isolated_reviewer_workspace export \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" --reviewer 1 \
  --destination '<approved-private-staging-parent>/reviewer-one'
"$PY" -m src.isolated_reviewer_workspace export \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" --reviewer 2 \
  --destination '<approved-private-staging-parent>/reviewer-two'
```

3. Through an approved privileged copy/transfer procedure, place the separate bundles into **distinct accounts' own private directories** outside the original shared 0700 parent. Set whole destination directory and batches to that human's unique UID and 0700/0600 modes. Ensure both accounts lack read/traverse privileges to the original canonical model key and to one another's workspace. Do not chmod the canonical key or shared parent to grant access.
4. Each human uses a local interactive TTY under THEIR OWN non-root distinct Unix UID, from a Python installation they can access. Their first-stage tool rejects execution if the workspace is still owned by the exporter, by another UID, or by root:

```bash
"$PY" -m src.isolated_reviewer_workspace status --workspace '<private-workspace>'
"$PY" -m src.isolated_reviewer_workspace review \
  --workspace '<private-workspace>' --reviewer-id '<the-real-human-private-id>'
```

5. Before any central import, the administrator runs the **read-only** source and return audit for each distinct-owner completed stage-one workspace. It checks all 1350 human field completions, exactly the 15 original source batches, all original prompt/response text, private file permissions and distinct non-root UID. This check explicitly DOES NOT verify real-world personhood, perform a file import, release benchmark labels, or authorize model publication:

```bash
"$PY" -m src.isolated_reviewer_workspace return-audit \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --workspace '<private-completed-workspace>'
```

## Two-workspace return gate (new)

After two actual human-operated, distinct-owner accounts complete stage one, the **read-only** pair audit checks the original model-key owner is distinct from both reviewer workspace owners, their source texts and all 1350 first-stage ratings are complete, the two reviewer slots differ, and the two Unix UIDs and private reviewer ID strings differ:

```bash
"$PY" -m src.isolated_reviewer_workspace return-pair-audit \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --first '<approved-human-one-completed-workspace>' \
  --second '<approved-human-two-completed-workspace>'
```

The pair check **does not** import anything, prove genuine two-person independence, reveal the model key, authorize stage two, or release a checkpoint. Different owner UIDs and different reviewer ID strings are necessary evidence, not sufficient human identity attestation. Synthetic tests explicitly simulate UID separation; the real fleet has NOT passed this gate.

## Remaining blockers / acceptance requirements

The actual approved transfer, verified two-person attestation, side-by-side access denial tests, independent return/import into the canonical adjudication engine, stage-two safe-only helpfulness handoff in the isolated accounts, and third-human identity separation are **NOT implemented or verified**. The return audit checks per-workspace content integrity and ownership, but ownership and a self-reported reviewer ID do not prove the existence of two independent humans. Full issue #107 should remain open until operational access, independent attestations, secure import, and access denial tests are confirmed.

Tests: `tests/test_isolated_reviewer_workspace.py` uses synthetic A/B/C prompt/response content only; checks key-free export, exact manifest, 0600/0700 privacy, no overwrite, tamper/symlink/hardlink rejection, first-stage editing/resumption, same-account execution denial, source integrity, completed return validation (synthetically simulated different UID), and reviewer-ID takeover refusal. No real model prompts, response texts, human grade data, or passwords have been exported.

**Release gate:** PR #103 remains draft; storage issue #106 remains open; no external benchmark behavior or checkpoint release is authorized.

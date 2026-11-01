# LFM3 reviewer return quarantine — private phase-one custody (2026-11-01)

**Implemented, tested with synthetic human annotations only. NOT run on real model response packets.** The original 450 XSTest prompts × A/B/C frozen generations and original reviewer packets remain untouched. No genuine person has reviewed 1,350 items per slot. Neither Unix UID separation alone nor a reviewer-ID string attests two independent humans.

## Goal and non-goals

`src.reviewer_return_quarantine.py` takes **two complete owner-separated, source-verified reviewer returns**, stores their phase-one human-only ratings in a **new owner-only custody quarantine**, and records a private checksum receipt written **after** the payload. It does NOT modify the original reviewer batches or model identity mapping; does not copy any model key, safe/unsafe labels, training weights or unrelated files; does not perform actual third-human adjudication or release a model. It never claims phase-two helpfulness or verified human independence.

Only do this after issue #107 administrator-provisioned two genuinely independent human-operated accounts, separate key-free workspaces and a secure approved return path. The code enforces the distinct non-root Unix ownership precondition, but **independent real-world humans require separate trusted evidence**, not generated signatures or text.

## Private operator workflow

Original sources and both reviewer workspaces must be accessible to an authorized custodian in a private shell. Choose a NEW output under an **existing owner-only 0700 parent**, outside the original source packet, canonical batch directory, and either reviewer workspace.

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Hypothetical approved paths below — do not run before actual human review.
"$PY" -m src.reviewer_return_quarantine stage \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --first '<approved-separate-human-1-workspace>' \
  --second '<approved-separate-human-2-workspace>' \
  --output-dir '<private-custody-parent>/phase-one-quarantine'

# Every verification must read the original canonical batches again:
"$PY" -m src.reviewer_return_quarantine verify \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --output-dir '<private-custody-parent>/phase-one-quarantine'
```

## Evidence layout

`phase-one-quarantine/` (0700) contains a private `private_quarantine_receipt.json` (0600) and `batches/` (0700). The latter contains the unmodified `private_batch_manifest.json` and exactly `reviewer_1`/`reviewer_2` (0700), each with 15 owner-only (0600) reviewed CSVs. There is **no model identity key** or answer-key copy. The receipt pins the hash of the original batch manifest and SHA-256 of all 30 returned private CSV files and explicitly marks all review-identity, human-independence, phase-two, adjudication, canonical-import and release claims FALSE.

## Release/security admission

Before writing, the function rechecks the original canonical custody, original model key hash, original source texts, 15 fixed batches per reviewer, 1,350 filled first-stage refusal/actionability responses per reviewer, empty benign helpfulness values, unique reviewer IDs, one return from each slot, and two distinct non-root Unix owners different from the model key owner. The original key is **never copied**.

Every `verify` requires BOTH original `--packet-dir` and `--batches-dir`: it re-audits canonical source integrity and manifest, compares each original ID/prompt/model response to the quarantined copy, recomputes all 30 sealed checksums, checks owner-only permissions and exact filenames, and verifies all 2,700 initial human rating fields + exactly one distinct reviewer ID per assignment. Changes to original batches, forged/resealed receipts with blank grades, added files, or an interrupted write without the final receipt are rejected.

An interruption may leave an *incomplete* 0700 output directory; this does not pass verification and is intentionally not overwritten. Administrative cleanup/retry must be separately approved. Do not use a failed or unsealed directory in downstream human analysis.

## Remaining work

- Real separate-account setup, physical/cross-account access tests and independent human attestation: **NOT performed**.
- Completion of two 1,350-item actual human review assignments: **NOT performed**.
- Secure, separately isolated second-stage 750 benign-helpfulness ratings for each reviewer: **NOT implemented here**.
- Independent third-human adjudication on actual entries and approved analysis: **NOT performed**.
- Import into original canonical review batches: **intentionally NOT implemented or authorized by this quarantine**.

Test evidence: `tests/test_reviewer_return_quarantine.py` uses 100% synthetic responses/ratings with mocked different Unix ownership. It checks valid staging, same-UID denial, canonical-file unchanged, receipt tampering, missing receipt, changed original source after seal, safe/unsafe stage-one semantics, and the principle that 750 synthetic benign entries remain ungraded per reviewer. These mocked UIDs are NOT a test of actual cross-account access.

Keep P0 issue #107 and storage issue #106 open. Draft PR #103 and frozen A/B/C model checkpoints remain unchanged; no model release is authorized.

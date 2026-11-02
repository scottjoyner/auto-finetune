# LFM3 — provenance-bound third-human adjudication handoff

**Status: implemented in draft PR #103 and tested using synthetic ratings only. Not deployed to real human reviewer accounts.** Independent personhood is not proven by a Unix UID. P0 issue #107 stays open; no checkpoint release is authorized.

## What is implemented

`src.verified_adjudication_handoff` verifies the new noncanonical, two-stage, private human review merge against the frozen original packets and canonical batches. It recomputes all disagreement/uncertain items from both human review sheets and prepares a **new** owner-only custody directory with two components: a private custodian receipt and a key-free `third_human_staging/` workspace containing only `blinded_adjudication.csv` plus `isolated_third_human_manifest.json`.

The candidate sheet contains only original blinded ID/prompt/model response; all third-human fields start blank. The workspace includes no model arm mapping, benchmark-safe/unsafe label, two original reviewers' judgments or IDs, model weights or source key. The private custody manifest pins the sealed two-stage merge receipt, exactly the selected source context and the two source reviewer **Unix UIDs**. None of the claims of actual human independence, completed helpfulness review, final consensus or release may be asserted by either manifest.

The custodian must supply the original **complete phase-one reviewer workspaces** along with the merged review package. The tool verifies those two originals still belong to distinct non-root Unix accounts different from the original key owner and binds those UIDs to the adjudication handoff. The third-human terminal refuses either original reviewer UID, the custodian/exporter UID, root, or an unrelated process UID. The separate real reviewer still must be identified and provisioned through approved administrator operations; a distinct UID is necessary but not sufficient for a genuinely independent human.

## Custodian commands (do not run on current ungraded original packets)

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
CANONICAL=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Only after separate humans have completed stage one and benign-helpfulness stage two:
"$PY" -m src.verified_adjudication_handoff prepare \
  --packet-dir "$PACKET" --canonical-batches "$CANONICAL" \
  --merge-dir '<approved-private-sealed-two-stage-merge>' \
  --first-a '<original-human-1-completed-stage-one-workspace>' \
  --first-b '<original-human-2-completed-stage-one-workspace>' \
  --destination '<approved-private-custodian-parent>/third-human-escrow'

# Custodian audit: revalidates ALL original evidence and blinded selected rows.
"$PY" -m src.verified_adjudication_handoff verify \
  --packet-dir "$PACKET" --canonical-batches "$CANONICAL" \
  --merge-dir '<approved-private-sealed-two-stage-merge>' \
  --first-a '<original-human-1-completed-stage-one-workspace>' \
  --first-b '<original-human-2-completed-stage-one-workspace>' \
  --destination '<approved-private-custodian-parent>/third-human-escrow' \
  --workspace '<approved-private-third-human-return-workspace>'
```

An approved administrator must transfer ONLY the two-file blinded staging bundle into a separately owned non-root reviewer directory, with both entries owner-only, and must confirm this human has **no access** to the other reviewers' workspaces or the model key. Do not copy the custodian receipt, source mapping, original grader IDs or original private merge package to the human. Do not loosen original folder ACLs.

## Actual independent third-human terminal

The genuinely independent third adjudicator runs only their own accessible local workspace; no canonical packet or custodian escrow path is required:

```bash
"$PY" -m src.verified_adjudication_handoff review \
  --workspace '<third-human-owned-isolated-workspace>' \
  --adjudicator-id '<authorized-third-human-private-id>'
```

The terminal uses an interactive human-only TTY, saves each first-stage refusal/actionability judgment atomically with no automatic labels, and never asks for helpfulness yet. The custodian can pass the completed workspace to `verify` again. This return verifier recomputes the dispute selection from the still-sealed original two-stage merge, ensures model text and selected IDs/order remain immutable, rejects reviewer identity/UID reuse, any premature helpfulness score, added files, missing ratings, altered receipt/source or changed original Unix-owner ledger, and reports completion as an aggregate only.

## Critical remaining gates

1. The third human's **benchmark-benign-only** adjudication helpfulness stage is still required before any behavioral result can be finalized. No safe/unsafe membership is revealed in this first-stage handoff.
2. The existing `reviewer_adjudication.resolve` cannot be used on the originally blank canonical reviewer batches as if they contained the independent merged results. A future approved, provenance-aware resolution step must accept the *sealed* two-stage source and a separately validated third human full first/second-stage return. No file substitution or model-unblinding shortcuts.
3. An independent trusted operational record must establish that the two original reviewers and the adjudicator are **three distinct actual human persons**, and real cross-account and model-key denial tests must pass.
4. No genuine human grades currently exist in either original 1,350-row reviewer assignment. The preparation/return tests simulate human ratings and use mocked UID ownership. These tests do not establish behavioral safety outcomes, consent, permission or release eligibility.

Keep issue #107 and storage #106 open; PR #103 remains draft and no model-release decision is authorized.

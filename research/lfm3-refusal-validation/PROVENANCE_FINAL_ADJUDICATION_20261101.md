# LFM3 XSTest — provenance-aware final adjudication and private statistical preflight

**Implemented as a draft, non-release research gate.** The real study still has 0/1350 human grades for each original reviewer, no verified independent human identity attestations, and no third-human work. The new source code and tests use synthetic annotations and simulated Unix ownership; they do **not** establish behavioral safety, scientific consensus, or release readiness.

## Purpose

`src/verified_final_adjudication.py` is deliberately separate from the legacy `reviewer_adjudication.resolve()` that expects human annotations in the **original canonical batch folders**. The original folders are kept blank and immutable under the isolated-reviewer custody model. Never substitute a quarantine copy into the original path.

The read-only resolver accepts the canonical source packet, canonical blank reviewer batches, a sealed noncanonical **two-stage merged** review package, and the TWO original reviewer-owned stage-one workspaces. It verifies the full original model-key/prompt/checkpoint mapping; all 30 sealed private review CSV files and source hashes; the 1350 original first-stage ratings in each separately owned reviewer workspace; and exact reviewer-ID, prompt, model-response, refusal, actionability, and uncertainty-note correspondence to the sealed copy.

If the original reviews agree on all applicable fields and neither is uncertain, the resolver uses that matching human judgment without requiring a third-human return. If a row is disputed, unclear, or uncertain, it rejects incomplete adjudication. It requires the **exact verified provenance-bound third-human** first-stage handoff and the matching **benign-only** third-human 0–3 helpfulness return, linked to the original UID ledger and frozen source. It must not accept a third adjudicator's ratings for undisputed rows. If the original batch tree has human correction requests not explicitly transported into the merged source, final resolution is blocked even if both reviewers agree.

The final table always contains exactly 450 prompts × 3 frozen model arms = 1350 validated pairs, with 250 benchmark-benign and 200 benchmark-unsafe source prompts. Benign rows have human 0–3 helpfulness; unsafe rows carry no such rating. The complete numeric table is validated against `reviewed_tradeoff_multiplicity._validate` before it can be passed to offline statistical code, including paired-prompt completeness and permitted binary/ordinal field ranges. Source/model mapping remains private; the CLI prints **aggregate custody status only** and never shows prompts, model responses, reviewer IDs, or case-level judgments.

## Read-only operator preflight

**Do not run on live ungraded packets expecting success.** The actual two human reviewer account handoffs, source data and scores must exist and be verified first.

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
CANONICAL=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

"$PY" -m src.verified_final_adjudication \
  --packet "$PACKET" --canonical "$CANONICAL" \
  --merged-dir '<approved-private-sealed-two-stage-merge>' \
  --first-a '<verified-original-reviewer-1-workspace>' \
  --first-b '<verified-original-reviewer-2-workspace>' \
  --escrow '<private-third-human-escrow>' \
  --phase-one '<completed-third-human-initial-review-workspace>' \
  --phase-two '<completed-third-human-benign-helpfulness-workspace>'
```

If there are **no** disputed/uncertain items, omit `--escrow`, `--phase-one`, and `--phase-two`. Continue to provide both original source reviewer-owned workspaces. Supplying third-human artifacts when none are justified fails closed.

This command is read-only; source data, human annotations, model checkpoints and canonical files remain unchanged. The return flag `source_evidence_verified=true` means only that the private *technical* custody and rating-field checks passed; it **does not** establish three real independent human reviewers.

## Statistical research boundary

The internal `resolve_private()` function returns an in-memory complete 1350-case numeric table, but only after the chain of source verification passes twice. `private_exploratory_diagnostics()` separately runs paired-prompt intervals and 12-contrast familywise multiplicity calculations **in memory only**, with at least 2,000 paired resamples. It writes nothing and is not the public report API. Every wrapper explicitly reports `independent_human_personhood_attested=false`, `human_rating_origin_independently_verified=false`, `public_scientific_claim_approved=false` and `production_release_approved=false`. Its paired output is relabeled as an identity-unverified provisional diagnostic, and `human_reviewed_evidence=false` prevents treating synthetic/test annotations as verified behavior.

If a correction request is pending, original reviewer input has changed since sealing, source IDs/orders differ, reviewer IDs were reused, third adjudicator original judgment fields are altered, an unsafe helpfulness score appears, private files are missing/tampered, or new conflicting annotations appear in canonical folders, **no numeric table or analysis passes admission**. The caller must not silently import, overwrite, publish, or unblind any evidence.

## Remaining P0 gates

Real separate non-root accounts for three genuine human participants; administrator-verified model-key and cross-reviewer access denial; independent identity attestation and authority/consent; actual blinded first-stage and benign-only ratings; correction/adjudication records; human statistical interpretation and external review. No one has provided these confirmations, and simulated UIDs/reviewer-ID strings cannot substitute for real humans. The storage recovery issue #106 remains open. P0 isolation/provenance issue #107 and PR #103 remain OPEN/DRAFT. **No checkpoint release authorized**.

Tests: `tests/test_verified_final_adjudication.py` uses synthetic source responses/labels and mocked UIDs. It covers 750 disputed benign entries, 600 undisputed unsafe entries, unanimous reviews with no third human, missing third-human return, missing helpfulness grade, altered original reviewer judgment, canonical source drift, untransported correction requests, uncertain third-human judgment, complete familywise preflight and explicit non-publication. No real reviewer data has been copied or edited.

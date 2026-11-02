# LFM3: private two-stage human-review merge — 2026-11-01

**Implemented and tested with synthetic human inputs; no actual human reviewer has yet entered any ratings.** This is a custody step, NOT proof of independent people or authorization to publish model-safety results. Keep draft PR #103, reviewer isolation P0 #107 and storage #106 open.

## Purpose

`src/isolated_two_stage_merge.py` combines both complete blinded stage-one reviewer workspaces (1,350 human refusal/actionability judgments per reviewer) with each SAME reviewer's complete private stage-two 750 benign-helpfulness scores. Output is a NEW owner-only 0700 directory with a `batches/` subtree formatted like the canonical 15×90 review batches and a **last-written** 0600 digest receipt. It contains the original batch manifest and the two reviewer CSV sets, but no `private_model_key.json`, benchmark ground-truth list, checkpoint weights, or human identity attestation.

**Counts in the synthetic acceptance contract:** 2,700 first-stage ratings and 1,500 additional benign helpfulness scores; each reviewer's 600 benchmark-unsafe responses remain without helpfulness. There is no model inference, automatic grading, altered original answer, or unblinding to human reviewers.

## Gated admission

Before any output directory is created the custodian must supply both phase-one workspaces, both phase-two workspaces, original owner-only packet and original canonical batch directory. The tool revalidates:

- exactly two source-complete 1,350-row reviewer returns, separate non-root Unix owners, distinct reviewer-ID strings and original model-key/source fingerprints;
- both 750-row private helpfulness returns from the **matching** original reviewer account, all grades human-entered 0–3 under the appropriate pinned benign-source eligibility;
- all original first-stage prompt, response, reviewer identity, refusal/actionability rating and uncertainty notes identical to the helpfulness handoff originals;
- exactly 250 benchmark-benign source prompts × 3 frozen model outputs; no unsafe grade or omitted benign item;
- canonical reviewer batch files completely BLANK, so there cannot be two silently competing rating sources;
- no output overwrite and no new output inside any original packet or reviewer workspace.

The merge then writes the output batches, hashes them and verifies all input files remain consistent across a **second, fresh read**. It writes the final private receipt **only after** all validation and flush; a process interruption or concurrent reviewer edit leaves an unsealed directory that fails the `verify` operation.

## Operator commands — DO NOT execute until two genuine humans finish review

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Custodian with approved access to BOTH complete human reviewer returns:
"$PY" -m src.isolated_two_stage_merge stage \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --stage1-a '<reviewer-1-completed-private-phase-one>' \
  --help-a '<reviewer-1-completed-private-benign-helpfulness>' \
  --stage1-b '<reviewer-2-completed-private-phase-one>' \
  --help-b '<reviewer-2-completed-private-benign-helpfulness>' \
  --destination '<approved-owner-only-custodian-parent>/two-stage-merged'

# Read-only verification must recheck the ORIGINAL canonical sources:
"$PY" -m src.isolated_two_stage_merge verify \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --destination '<approved-owner-only-custodian-parent>/two-stage-merged'
```

## Sealed custody and non-claims

The private digest receipt pins all 30 merged CSVs and the unchanged canonical manifest, and fixes: `human_independence_attested=false`, `third_human_adjudication_complete=false`, `original_canonical_modified=false`, `model_key_copied=false`, `release_authorized=false`. Each verification rechecks the real canonical source packet and all 30 original canonical batches, detects competing annotations, confirms exact blinded source text, SHA-256 of merged batches, all first-stage human field domains, 750 safe-class ratings per reviewer, unique reviewer IDs and remaining 600 unsafe no-helpfulness cases. Even if someone rewrites the local digest receipt to match an unsafe helpfulness substitution, the pinned source/key membership check refuses it.

A merged record marked `unclear` or `uncertain` requires a genuine third-human adjudicator. Two different reviewers can disagree even with all grade fields present. **This package never claims human consensus**, source-independent human identity, publication eligibility, or a release. It is deliberately separate from the frozen canonical packets. The existing third-human workflow needs an approved provenance-aware handoff before any behavioral result is inferred.

## Remaining real-world blockers

Actual separate human-operated OS identities and trustworthy identity attestations; independent key/cross-reviewer access-denial tests; real phase-one/phase-two ratings and approved secure transfers; provenance-aware third-human dispute/correction adjudication and analysis; storage issue #106. Existing reviewer isolation issue #107 stays OPEN/P0. No real reviewer folders, live model outputs, user accounts, human grades or checkpoint weights were changed by creating/testing this module.

Regression suite `tests/test_isolated_two_stage_merge.py` runs entirely on synthetic source/grade values with **mocked distinct Unix owners**; those tests do not prove real two-person independence. They cover source-safe merge, wrong/missing stage-two inputs, duplicate or swapped reviewer returns, tampered rating/hash, forged unsafe helpfulness with rewritten receipt, mid-merge mutation, canonical drift, missing receipt/extra files, and explicit non-authorization.

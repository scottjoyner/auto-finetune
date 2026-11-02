# LFM3 XSTest: independently isolated benign-helpfulness phase two — 2026-11-01

**Code is implemented and tested only with synthetic human responses. No actual reviewer has entered first-stage or second-stage scores, no reviewer OS accounts were provisioned, and no real private review text has been exported.** Draft PR #103 remains non-releasable; issue #107 remains P0 open.

## What phase two is and is not

The new `src/isolated_benign_helpfulness.py` creates a *separate, reviewer-specific, private* `benign_helpfulness.csv` of **750 benign model responses** for one reviewer, plus a source-integrity manifest. The 750 rows are the three frozen LFM3-1.7B model outputs for the pinned **250 benchmark-safe source prompts**. Original first-stage reviewer judgments (response/refusal, harmful actionability, optional notes) are copied intact, for context and source verification; the benign helpfulness 0–3 field starts blank. Source IDs, model arm mapping, the private model key, all 600 model outputs for the 200 unsafe source prompts, and the other human reviewer's work are **not** copied. The selection reveals only that a selected prompt belongs to the benchmark-benign subset, which is intentionally allowed solely *after* all 1,350 first-stage ratings are fixed.

**This is not the original shared-account stage-two tool.** It supports separate key-free reviewer accounts and reads that same reviewer's independently owned stage-one workspace without needing the central model key. The custodian/administrator may read the pinned source model mapping to produce and verify the fixed safe-only handoff but must not copy that mapping into the private reviewer workspace.

## Source security gates

The custodian must provide the original owner-only `--packet-dir`, unchanged `--batches-dir`, and a **complete, different-UID, non-root** 1,350-row `--phase-one` workspace. `first.verify_return()` re-audits all original prompts/responses, private permissions, and the complete phase-one annotations. An additional check rejects competing annotations in the original canonical packets. Only then does an export query the source-key mapping and pinned 450-case dataset for 750 safe-class items. A new 0700 destination contains **exactly two 0600 files**, no key, no other-reviewer material, and no unsafe rows.

The manifest pins the original reviewer source-sheet SHA, each of 15 immutable first-stage CSV byte hashes, ordered benign IDs, and each benign row's ID, prompt, model response, first-stage rater identity and first-stage judgments. It also records exporting UID and original human's UID with `human_identity_attested=false`, `model_key_copied=false`, `release_approved=false`.

## Operator procedure (NOT ready to run on live reviewer data)

Two real distinct human reviewer accounts and trusted private transfer must be configured beforehand. The reviewer must receive the phase-two workspace **under their SAME separate Unix account** as their phase-one work, not under the exporter account or as root. The operations team must independently verify actual distinct people and deny cross-reviewer/key access. The paths below are placeholders and must never be used without their actual authorized human owners:

```bash
cd /media/scott/SSD_4TB/knowledge/lfm3-direction-stability-20261101/research/lfm3-refusal-validation
PY=/home/scott/lfm3-research/train-venv/bin/python
PACKET=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_review_private
BATCHES=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_review_batched_private
export PYTHONDONTWRITEBYTECODE=1

# Authorized custodian ONLY, after verified original phase-one completion:
"$PY" -m src.isolated_benign_helpfulness export \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --phase-one '<human-one-private-completed-stage-one>' \
  --destination '<approved-private-handoff-parent>/human-one-benign-stage-two'

# Human reviewer ONLY, own non-root Unix login and interactive terminal:
"$PY" -m src.isolated_benign_helpfulness status \
  --workspace '<human-one-owned-benign-workspace>' \
  --phase-one '<human-one-private-completed-stage-one>'
"$PY" -m src.isolated_benign_helpfulness review \
  --workspace '<human-one-owned-benign-workspace>' \
  --phase-one '<human-one-private-completed-stage-one>' \
  --reviewer-id '<original-human-one-private-id>'

# Custodian ONLY, after the human has graded all 750 benign cases:
"$PY" -m src.isolated_benign_helpfulness verify-return \
  --packet-dir "$PACKET" --batches-dir "$BATCHES" \
  --phase-one '<human-one-private-completed-stage-one>' \
  --workspace '<human-one-owned-benign-workspace>'
```

Repeat independently for reviewer 2 under their **distinct** Unix UID and with their own private stage-one/phase-two directories. An account created or controlled only by an agent does NOT qualify as an independent human.

## Completion and correction discipline

Every human score is entered explicitly in a local TTY, atomically saved with restricted permissions, and never automatically generated. Already-entered scores are **append-only** in the ordinary terminal: an accidental score must go through an auditable correction and third-human adjudication process, not a silent overwrite. The same original reviewer-ID string must match the frozen first-stage records. A changed phase-one response/notes, any unexpected file, altered prompt/model output, change in ownership, or a non-benign item rejects verification. Rewriting a local 750-row manifest to smuggle in an unsafe source cannot pass central source-key membership checking. Canonical packets must remain blank to prevent competing evidence.

`verify-return` is **READ ONLY**: it checks all 750 human 0–3 helpfulness scores, eligibility against the original pinned benchmark and private source mapping, and frozen first-stage hashes. It does not copy scores back into the canonical batches, manufacture human identity attestation, run third-human adjudication, compute a behavioral claim, or authorize model release.

## Remaining gates

Separate human-operated OS accounts; administrator-approved secure transfers; actual non-root cross-account and model-key denial tests; independently witnessed human identity; both genuinely completed 1,350-item phase-one sets; both genuinely completed 750-item phase-two sets; secure import/merge of phase-two into a new quarantine; human adjudication and verified analysis; existing storage incident #106. These are all **not done** on the real review data. Issue #107 and draft PR #103 remain open.

Synthetic tests in `tests/test_isolated_benign_helpfulness.py` cover source eligibility 250×3, no unsafe/model-key/other reviewer export, incomplete first-stage rejection, ownership/runtime denial, immutable first-stage linkage, append-only human input, forged unsafe IDs with recomputed handoff hashes, competing canonical ratings, and complete stage-two return verification. Mocked Unix UID checks do not prove real OS access isolation or human independence.

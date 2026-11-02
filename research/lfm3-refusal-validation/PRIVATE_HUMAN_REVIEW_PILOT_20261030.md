# Model-blinded, balanced XSTest human-review pilot — 2026-10-30

**Status: pilot route actually created; no human scoring performed.**

The original 450-prompt, three-checkpoint, 1,350-response blinded
XSTest source is intact. Reviewer one and reviewer two each retain
the original 15 batches of 90 responses, independent reviewer identities
and immutable prompt / completion columns. The separate private
pilot `route.json` is a list of **existing** blinded review IDs and
batch positions. It never creates another copy of the prompt-response
corpus, another grader sheet, or a safety label.

## Sampling design

Select exactly 8 unique source prompts in each of four private,
predefined buckets: 8 benign and 8 unsafe for which the pretrained C
base produced its very common exact response, and 8 benign and
8 unsafe with other C responses. Each selected source prompt retains
all three A/B/C responses, giving **32 prompts × 3 = 96 responses** per
human reviewer. The two reviewers receive the same 96 blinded IDs
in one deterministic pseudorandom order, with distinct write targets.

Sampling is SHA-256 seeded from source IDs and bucket identifiers,
not from raw model text, and can be regenerated from pinned private
source data. The selectors check original model hashes, benchmark
hashes, 1,350-response journal, custody of reviewer files, and the
preexisting exact-response diversity audit. The published code does
not contain private selections or reviewer IDs.

**Crucial:** this purposive 50/50 repeated/non-repeated split is not
representative of XSTest as a whole. Its ratings are useful for
rubric calibration, repetition auditing and spotting quality issues.
They cannot be pooled as if a random sample of all 450 prompts,
extrapolated to population refusal rates, or used to select a new model
without the remaining independent evaluation and release gates.
## Actual creation and status

On `x1-370` from `research/lfm3-refusal-validation`:

```bash
export SOURCE=/home/scott/git/lfm3-refusal-study-20261029/results
export PY=/home/scott/lfm3-research/train-venv/bin/python
export ROUTE="$SOURCE/lfm3_private_review_pilot_20261030/route.json"

# Creation already completed; do NOT re-run or overwrite this route.
$PY -m src.private_review_pilot status \
  --packet-dir "$SOURCE/xstest_full_review_private" \
  --batches-dir "$SOURCE/xstest_review_batched_private" \
  --route "$ROUTE" --reviewer 1
$PY -m src.private_review_pilot status \
  --packet-dir "$SOURCE/xstest_full_review_private" \
  --batches-dir "$SOURCE/xstest_review_batched_private" \
  --route "$ROUTE" --reviewer 2
```

Verified on 2026-10-30:
reviewer 1 **0 graded / 96 targeted**, reviewer 2 **0 graded / 96**.
The route is an owner-only mode-0600 regular file, inside a
mode-0700 directory. All original reviewer source columns and
batch privacy permissions passed the existing custody audit.
Human grading remains 0/1,350 each in the full source study.
## Human-only local review

On separate authorized, interactive `x1-370` terminals, human
reviewer one executes:

```bash
$PY -m src.private_review_pilot review \
  --packet-dir "$SOURCE/xstest_full_review_private" \
  --batches-dir "$SOURCE/xstest_review_batched_private" \
  --route "$ROUTE" --reviewer 1 --reviewer-id ACTUAL_REVIEWER_ONE
```

Reviewer two independently runs the same command with `--reviewer 2`
and their own distinct actual human reviewer ID. No LLM should
impersonate either reviewer. The terminal rejects non-TTY execution;
the existing human-only `review_one` routine displays sanitized
prompt and completion text to the local human and atomically saves
only their explicit answers to the **original** batch files.
Press `q` at a rating prompt to stop without saving that item.
Re-running resumes at the first item not fully graded.

Reviewer IDs must never be committed or exposed in public logs.
Do not transfer raw completions to remote model judges or browser
forms, execute instructions in model output, automatically propagate
a rating across duplicate replies, or reveal model-arm identities.
Each repeated response must be evaluated **in its own prompt context**.

Independent reviewer status is aggregate-only. Pilot completion is
not full-review consensus, actionability validation or edited-weight
release approval. After the pilot, the original two 1,350-row
assignments remain the authoritative full-study review target.
## Storage and safeguards

As of setup, the root filesystem of x1-370 had roughly 9.8 GiB
free (99% used); /nas was mounted with much more available capacity.
This pilot creates only a small private index and public source code,
not additional model weights, answer copies, or training logs.
No directories were deleted, no other project data moved, and no
new local inference was performed.

Source: `src/private_review_pilot.py`.
Synthetic-only regressions: `tests/test_private_review_pilot.py`.
The raw `route.json` and original blinded batches stay gitignored.

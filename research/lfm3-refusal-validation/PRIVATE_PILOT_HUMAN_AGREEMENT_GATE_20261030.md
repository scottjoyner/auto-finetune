# Human-reviewed LFM3 pilot: post-rating agreement gate (2026-10-30)

The model-blinded 32-prompt/96-response-per-reviewer pilot is selected
and independently source-attested. This module is the **next step AFTER**
two distinct real humans complete the selected responses in their
original private 15x90 CSV batches. It is NOT an automatic grader,
not an independent human substitute and not a model-selection score.

## Real-machine outcome on x1-370

On the original private 0/96 + 0/96 pilot, the new
`src.private_pilot_agreement` command returned exit **2** with
`pilot_human_agreement_ready=false`, no safety approval, and **created
no report**. The original frozen responses, private route, reviewer
source data, and model checkpoints were not modified. Synthetic-only
tests demonstrate that 96 fully graded human pairs would be accepted;
no fabricated grades were written into the actual benchmark.

## Full-completion requirements

The reporter requires: the exact source-verified pilot route,
intact original 1,350-record journal and model/checkpoint hashes,
intact original reviewer batch files, three distinct A/B/C answers for
each of 32 selected prompts, exactly 48 benign and 48 unsafe
blinded responses, and 96 matching grades in both reviewer assignments.

It rejects any missing, uncertain, inconsistent, or unpaired grades,
missing benign helpfulness, spurious unsafe helpfulness, duplicate
checkpoints, or reuse of the same human identity across reviewers.
It publishes neither model-arm names nor model-specific results.
When complete, it calculates exact agreement and Cohen's kappa on
response category (96 pairs), harmful actionability (96 pairs), and
quadratic-weighted helpfulness (48 benign pairs). An undefined kappa
under degenerate category marginals remains **null**, not 1.0.
Disagreement counts require human adjudication; the tool never
rewrites human votes or silently resolves conflict.
## Command (requires genuine completed human reviews)

Run locally from `research/lfm3-refusal-validation`:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
BASE=/home/scott/git/lfm3-refusal-study-20261029/results

$PY -m src.private_pilot_agreement \
  --results-dir "$BASE/xstest_full_20261029_v1" \
  --packet-dir "$BASE/xstest_full_review_private" \
  --batches-dir "$BASE/xstest_review_batched_private" \
  --route "$BASE/lfm3_private_review_pilot_20261030/route.json" \
  --output "$BASE/lfm3_private_review_pilot_20261030/private_agreement.json"
```

If human review is incomplete the process exits **2**, without creating
the private report. On success the output is exclusive-create, mode 0600
and stored only within the mode-0700 local private pilot directory;
it contains aggregate agreement and no raw prompt, response, blinded
record ID, human reviewer identity, or model assignment.

**Caution:** this intentionally over-samples cases where the pretrained
base repeats its modal response. The 96 responses are NOT a random
population sample, so agreement does not estimate a general refusal
rate. The full 1,350-response review by two independent humans, separate
actionability assessment and risk review remain required before any
scientific model-level safety claim or checkpoint release.

Disk on x1-370 was critically low (~3.5 GiB free during testing).
No additional model runs or checkpoint copies were initiated.

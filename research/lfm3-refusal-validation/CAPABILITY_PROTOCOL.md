# Independent capability-retention axis for the three fixed LFM3 checkpoints

The main refusal study retains exactly three model checkpoints:
A original instruction-tuned, B M1-edited A, and C pretrained base.
Capability evaluations are independent measurements, NOT additional models.

## Public ARC-Challenge source

- Source: `allenai/ai2_arc`, ARC-Challenge test parquet.
- Pinned HF revision: `210d026faf9955653af8916fad021475a3f00453`.
- SHA-256 of test parquet: `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9`.
- Total questions: 1,172; raw questions and answers remain local.

## Experimental ARC-style probe

The runner `src.arc_capability_eval` selects the same seeded sample for A/B/C,
formats each question with canonical A-E option letters, and compares the next-token
letter logits in a **raw completion prompt** rather than using a chat template. This
is intentional to reduce a major fairness mismatch for the pretrained base.

The metric is **only a one-token letter-completion accuracy proxy**. It is *not*
the benchmark's official choice-sequence conditional likelihood score, does not
use multiple option permutations, and cannot stand alone as a capability-retention
claim. A full publishable study will use the test's complete set, multiple choice
order permutations, proper normalized conditional likelihood, and standardized
benchmarks of instruction following and tool use.

Example local command:

```bash
cd /home/scott/git/lfm3-refusal-study-20261029
/home/scott/lfm3-research/train-venv/bin/python -m src.arc_capability_eval --sample-size 12 --seed 17
```

Private case-level outputs are never committed; only sanitized denominator
counts and limitations may appear in the paper or pull request. Never infer
that successful reasoning implies resistance to malicious or unauthorized actions.


## Preliminary 12-item smoke result (not official accuracy)

A original: 4/12; B M1-edited: 4/12; C pretrained: 8/12.
All twelve A/B raw-completion selected letter predictions were identical.
This cannot validate preservation of capability; report only as an
integration test. No benchmark question or output text is public.

## Full pinned ARC-Challenge conditional-choice likelihood study

The old 12-question experiment only measured one-token option-letter logits.
The new evaluator, src.arc_full_likelihood_eval, instead computes
the conditional loglikelihood of the entire answer-choice text after a
fixed raw prompt of the form "Question: <question> newline Answer:".
It records both the summed token loglikelihood and mean-per-token
loglikelihood choice accuracy for each checkpoint separately.
All prompts, target answers and scored cases remain private.

The evaluation uses the same three hashes as the XSTest benchmark,
the pinned ARC-Challenge test split of 1,172 questions, fixed answer
order, BF16 CPU inference and no real tools. A private JSONL journal
with fsync permits resuming work. Its run manifest refuses mismatched
models or dataset content. A new output directory is required for
different sample sizes.

Pilot:
    python -m src.arc_full_likelihood_eval --sample-size 3 --results-dir results/arc_choice_likelihood_pilot_private

Full study:
    python -m src.arc_full_likelihood_eval --sample-size 0 --results-dir results/arc_choice_likelihood_full_private

This is a **custom full-test likelihood protocol**, NOT an official
lm-evaluation-harness benchmark score and not a validated safety
measure. Confirmation with a standardized independent harness,
choice-order permutations, and separate instruction following
remains required for strong capability-retention claims.

## Auditing the running full ARC likelihood journal

The independent read-only audit checks the exact three-arm manifest,
pinned source checksum and revision, full selected case order, expected
answer indices, finite log-probability scores, candidate token lengths,
predicted-index consistency, boolean correctness and contiguous journal
order. It does not report partial score or safety estimates.

    python -m src.arc_likelihood_progress_audit \
      --results-dir results/arc_choice_likelihood_full_private \
      --verify-live-weights

The --verify-live-weights option rehashes the actual three local weights
and fails if any differs from the original run manifest. Omitting it
verifies journal metadata only, and is faster for repeated read-only
progress checks.

The full run targets 1,172 questions for each of A/B/C, totaling
3,516 scored questions. Only a completed run should be interpreted as a
matched three-arm capability comparison, with a custom-scoring disclaimer.

A supplementary inference parity check on two real ARC cases on the
original LFM3 checkpoint compared batched, right-padded answer-option
log-likelihoods with independent unpadded scoring. The largest observed
absolute score difference was below 5e-7 log-probability units.
This is an implementation smoke check, not a standardized benchmark.


## Completed paired three-arm capability comparison (2026-10-29)

All 3,516 custom likelihood scores passed the independent journal audit.
Length-normalized accuracy: A 315/1,172 (26.88%), B 301/1,172
(25.68%), C 493/1,172 (42.06%). Matched B-A change: -1.19 percentage
points, with 14 gained and 28 lost items and a seeded paired-bootstrap
95% interval of [-2.30, -0.09] percentage points. As a sensitivity
check, unnormalized likelihood yielded 277, 263 and 459 correct
respectively. These are custom raw-completion results, not official
ARC scores or evidence of generally preserved capabilities.

The full-case pairing and bootstrap are implemented in
src/arc_likelihood_paired_diagnostics.py; the sanitized aggregate
is stored under evidence/20261029_full_arc_paired_capability_diagnostics.json.
No further checkpoints are part of this experiment. Two-human-reviewer
XSTest outcomes and standardized independent capability confirmation
remain pending.

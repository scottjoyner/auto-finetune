# Matched-prompt latency audit conditioned on C's repeated answer

**Date:** October 31, 2026. **Study:** frozen full 450-prompt XSTest,
three pinned LFM3-1.7B checkpoints, 1,350 private CPU BF16 responses.
**No new model inference, training, weight edits, or human grades.**

The first latency study found that the pretrained C checkpoint's
unsafe median elapsed time (2.6165 s) is unusually low. C emitted one
identical short keyword-positive answer 156 times (29 benign, 127 unsafe).
A second audit now reuses the **same prompt IDs** to contrast C's
recorded elapsed time against A's and B's, separately for those where C
returned the modal repeated answer and those where it did not.

## Primary matched-prompt results

A minus sign in C-minus-A or C-minus-B means C completed faster.
Elapsed seconds include prefill, generation and surrounding overhead;
they are not decode tok/s.

| XSTest class; C response subgroup | n | C faster / slower than A | Median C minus A | C faster / slower than B | Median C minus B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Benign; repeated C mode | 29 | 26 / 3 | -7.810 s | 29 / 0 | -13.139 s |
| Benign; other C responses | 221 | 45 / 176 | +0.607 s | 140 / 81 | -0.179 s |
| Unsafe; repeated C mode | 127 | 126 / 1 | -8.550 s | 127 / 0 | -13.021 s |
| Unsafe; other C responses | 73 | 32 / 41 | +0.165 s | 45 / 28 | -0.356 s |

Thus C's apparent unsafe elapsed-time advantage is concentrated in
the requests on which it produced the same brief response. Without
that repeated output, the recorded C-versus-A unsafe difference is
close to zero and the sign direction slightly favors A (41 vs 32).
The Holm-adjusted exploratory two-sided sign test for this subgroup
is **p approximately 0.349**, which cannot support a claim of C's
speed superiority. In benign nonmodal output contexts, C was
slower than A on **176 of 221** matched prompts.

These findings are a strong **confounding diagnostic**, but the
grouping is defined using C's output itself. It is post-hoc
selection on the outcome, not a randomized or externally assigned
experimental condition. The results cannot isolate model architecture,
compute efficiency, causal latency advantages, or actual throughput.
## Research integrity

The analyzer uses `src/xstest_latency_repetition.summarize` to
revalidate the complete matched source panel, type, class, exact
recorded refusal-keyword flag and positive finite elapsed times.
It also independently checks that C's original 156-copy mode is
preserved (29 benign, 127 unsafe). An eight-contrast family of
exploratory two-sided paired sign tests is Holm-adjusted.
This statistical treatment helps prevent over-reading the post-hoc
comparisons; it **does not repair the conditioning bias**.

The CLI additionally verifies the pinned benchmark SHA-256, local
three-checkpoint SHA-256 identities, and the private original
1,350-generation journal digest and completeness. Public output
contains only aggregate count/timing statistics and provenance
hashes. There are no raw prompts, generated answer strings, private
response IDs, reviewer identities or per-response digests.

The frozen run has no per-token decoding timestamps, true output
token counts, finish reason, randomized run order, or controlled
concurrency. A whitespace word count is reported only as text-length
context and is **not** a token-throughput measurement. The pretrained
C is not an instruction-tuned checkpoint; comparative behavior is
not comparable to safety refusal efficacy without independent review.

The two human full-study reviewers remain 0/1,350 graded each.
Neither the automated lexical proxy nor these runtime measurements
constitutes human behavioral refusal, harmful actionability,
benign helpfulness or permission to release a weight edit.

## Reproduction

From `research/lfm3-refusal-validation` on authorized `x1-370`:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
RUN=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1

$PY -m src.xstest_modal_paired_latency \
  --results-dir "$RUN" \
  --out evidence/NEW_UNIQUE_modal_paired_latency.json
$PY -m pytest -o addopts= -q tests
```

The checked-in immutable aggregate is
`evidence/20261031_xstest_modal_paired_latency.json`.
The writer refuses to overwrite an existing aggregate.
No weight-bearing job is invoked or benchmark source modified.

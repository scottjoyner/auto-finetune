# XSTest response repetition and elapsed-time confounding — 2026-10-31

**Status:** Reproducible analysis of the original pinned three-checkpoint
450-prompt XSTest study: 1,350 real, completed CPU BF16 generations.
No new inference, model edits, private text publication or human grading.

## Full observed elapsed-time result

These are median **per-generation elapsed seconds** from the original
journal, including prompt processing, output and runtime overhead.
They are NOT true generated tokens per second, inference efficiency,
energy, same-length latency or hardware-agnostic comparisons.

| Frozen checkpoint | Benign median elapsed seconds (250) | Unsafe median elapsed seconds (200) |
|---|---:|---:|
| A original instruction-tuned | 14.9825 | 11.7500 |
| B M1 weight-edited | 15.8860 | 15.7000 |
| C pretrained base | 15.5300 | 2.6165 |

C's **same exact short, keyword-positive string** occurs 156 times
across the corpus, split between 29 benign and 127 unsafe prompts.
The original private response remains unpublished.

| C group | Number of responses | Median elapsed seconds |
|---|---:|---:|
| Benign: modal repeated string | 29 | 2.514 |
| Benign: all other outputs | 221 | 15.620 |
| Unsafe: modal repeated string | 127 | 2.523 |
| Unsafe: all other outputs | 73 | 15.456 |

This gives a plausible explanation for C's low *aggregate unsafe*
response time: a majority of those outputs are brief, repeated stock
answers. The repeated-versus-other grouping is **post hoc**,
not randomized or matched, so these figures establish neither causality
nor an advantage in sustained inference throughput.
## Matched-prompt timing-direction comparisons

For each prompt, subtract the original recorded elapsed seconds of
two model conditions. Report the median difference and conduct
exploratory exact paired sign tests, Holm-adjusted over six planned
post-hoc contrasts (three pairs × two dataset strata).

| Matched contrast | Benign median delta, s | Unsafe median delta, s |
|---|---:|---:|
| B minus A | +1.0070 | +2.3415 |
| C minus A | +0.4660 | -6.8225 |
| C minus B | -0.3055 | -9.0855 |

The six Holm-adjusted p-values are recorded in the JSON evidence.
The comparisons were not preregistered as inference-speed tests;
runs were sequential, not randomized or interleaved. Since output
length differs dramatically across arms and requests, a pairwise
elapsed difference does NOT isolate architecture or kernel speed.

The full run's journal stores only `elapsed_seconds`,
`response` and the original weak refusal-keyword indicator,
not generated-token counts, token timestamps, stop reasons,
prefill duration, per-token decode duration, energy, throughput,
or concurrency. Whitespace-word counts are not tokenizer tokens.
Any future true throughput benchmark must instrument those separately.

## Integrity and publication limits

The analyzer validates all 450 unique source IDs and 1,350 matched
arm/case pairs, safe/unsafe strata, response schema, complete private
journal digest and frozen checkpoint SHA-256. It recomputes refusal
keyword flags to detect source drift and separately enforces the
single 156-count C mode. It emits only aggregate timings, matched
comparison counts and hashes, without private case IDs, prompts,
model-generated text or per-response fingerprints.

**Crucial:** lower response latency and a refusal-like repeated phrase
cannot prove refusal quality, harmful-actionability resistance,
helpfulness, or model readiness. Human graders remain at zero graded
responses in the separate 1,350-case review assignments. No edited
model is approved for release or production.
## Reproduce

From `research/lfm3-refusal-validation` using the existing pinned
local Python environment, on the authorized x1-370 node:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
ORIGINAL=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1

$PY -m src.xstest_latency_repetition \
  --results-dir "$ORIGINAL" \
  --out evidence/NEW_UNIQUE_latency_repetition.json
$PY -m pytest -o addopts= -q tests
```

The public aggregate is
`evidence/20261031_xstest_latency_repetition.json`.
The tool refuses to overwrite an existing result. All raw answers
and the original case-mapping journals remain private.

Storage note: x1-370 remains near 100% root filesystem utilization.
No new model inference, weight copying, or cleanup was conducted
to create this audit; the existing disk preflight still blocks large
writes until capacity is safely restored.

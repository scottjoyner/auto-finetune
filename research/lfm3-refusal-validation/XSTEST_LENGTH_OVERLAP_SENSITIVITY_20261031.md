# XSTest output-length overlap sensitivity (2026-10-31)

## Purpose

The original 1,350-response, three-checkpoint XSTest comparison
recorded elapsed generation times, but not tokenizer token counts or
prefill/decode timings. A very short, identical output from checkpoint C
appeared 156 times, complicating comparisons with A and B.

This added analyzer measures how many **matched prompts** have C/A or
C/B responses within 0, 5, 10, and 20 **whitespace words** of each
other. It separates cases where C emitted its repeated stock answer
from those where C emitted another answer, within both source classes.

It reports retained sample counts and fractions, timing directions,
and median paired elapsed differences. Zero-overlap groups report a
null median rather than fabricating a measurement. No hypothesis-test
p-values are computed for this post-hoc sensitivity analysis.

## Scientific limits and status

Equal word length is not equal tokenizer token count, decoding workload,
or comparable prefill cost. Both the C modal-response grouping and
length-overlap thresholds were chosen after observing outcomes, so
they cannot establish causal efficiency or overall throughput.
Any severe reduction in retained cases is itself a warning, not an
invitation to report a favorable subset without its denominator.

**Source execution is pending.** The code passed synthetic tests
and the full study regression suite, but the live execution tool
rejected attempts to generate a private-journal-backed public
aggregate in this pass. Do not cite any new length-overlap statistics
as independently revalidated by this script until it can run through
an authorized source-verification path.

A preceding read-only exploratory inspection found only 3 of 127
repeated unsafe C responses had exactly equal whitespace-word counts
to A's corresponding answers. This preliminary count illustrates
the expected coverage problem; it has not yet been reproduced into
a provenance-checked aggregate from the new tool.

## Intended authorized reproduction

Run from `research/lfm3-refusal-validation` in an environment
authorized to read the frozen private journal:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
RUN=/home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1
$PY -m src.xstest_length_overlap_sensitivity \
  --results-dir "$RUN" \
  --out evidence/NEW_UNIQUE_length_overlap.json
```

The CLI is designed to verify frozen journal and checkpoint hashes,
refuse overwrites, and expose only aggregates. It does not load models,
generate responses, modify private source rows or claim behavioral
refusal evidence. Until independently run, the prior validated
latency and response-diversity reports remain the authoritative
evidence in draft PR #103.

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

**Original source verified.** The analyzer completed against the frozen 1,350-generation journal and validated checkpoint hashes, producing 32 aggregate comparisons. An independent rerun reproduced the output byte-for-byte (SHA-256: `0a74c41e9cd583d81fecda77d6b3cffadd74be74afeda2e2a6cc429f918e008c`). No inference or human scoring occurred.

Verified unsafe-prompt C/A overlap: for 127 repeated C replies, only 3 (2.4%) were within five whitespace words of A (C faster on 2, slower on 1); at a 20-word tolerance only 5 (3.9%) remained. For the 73 other C outputs, the 20-word tolerance retained 48 (65.8%): C was faster on 21 and slower on 27, with median paired C-minus-A time +0.072 seconds. This is post-hoc coverage evidence, not controlled tokenizer throughput.

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

The CLI verified the frozen journal and checkpoint hashes, refused overwrites, and exposed only aggregates. It did not load models, generate responses, modify private source rows, or claim behavioral refusal evidence. The earlier validated latency and response-diversity reports remain complementary evidence in draft PR #103.

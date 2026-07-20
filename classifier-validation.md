# Classifier validation (hand-labeled sample)

`analyze` classifies every session with a deterministic heuristic
(`classify_bucket` / `classify_difficulty`) and flags failures via
`_is_error` on tool outputs. This checks how trustworthy that is, because
**strata weighting, the verify pass-rate, and the 27 contrastive repair
pairs all assume the heuristics are right.**

## Method

- `python -m src.cli validate-classifier` builds a stratified label
  sheet (`src/validate_classifier.build_sheet`): 15 sessions from
  `failures.jsonl` (pred `is_error=True`) + 15 random non-failures,
  each with the heuristic prediction + a compact, human-readable view
  (first user request, tool-action trace, error snippet).
- The sheet was hand-labeled for `true_bucket` / `true_is_error` /
  `true_difficulty`, then `score()` reports metrics.

Sample = 30 sessions from the live cleaned corpus (15 fail / 15 ok).

## Results

| metric | value |
| --- | --- |
| bucket accuracy | **0.50** |
| error detection precision | **0.53** |
| error detection recall | **1.00** (tp=8, fp=7, fn=0, tn=15) |
| difficulty accuracy | **0.97** |

Per-bucket (precision / recall / support):

| bucket | P | R | n |
| --- | --- | --- | --- |
| debug | 0.29 | 1.00 | 4 |
| reasoning | 0.71 | 0.71 | 14 |
| code-search | 1.00 | 0.50 | 2 |
| shell / multi-file-refactor / docs / mixed | 0.00 | 0.00 | 5+3+1+1 |

## Findings

1. **Error markers over-fire on benign text (precision 0.53).** Every
   real error is caught (recall 1.00, no false negatives in the sample),
   but ~47% of flagged "failures" are false positives: the marker matched
   inside a *file read* (`<path>...</path><content>...error...</content>`),
   a knowledge-harvest startup banner, or a cron health-summary JSON.
   So `failures.jsonl` (1338) is heavily contaminated, and the
   `debug` bucket (which keys off `has_error`) is inflated.

2. **`debug` is massively over-predicted (precision 0.29).** The
   cascade rule `(debug intent) OR (has_error AND shell>0) -> debug`
   pushes half of "debug" predictions into actually-implement /
   refactor / shell / cron-automation tasks. Root cause is the error
   false-positive above.

3. **Buckets beyond debug/reasoning are barely emitted.** In this
   sample the classifier produced only `debug` and `reasoning`; real
   `shell` / `multi-file-refactor` / `docs` / `mixed` sessions were
   all misrouted (precision 0.00). Caveat: the 30-sample is
   skewed toward failures + cron, so this is partly sampling — but the
   early `debug`/`reasoning` rules clearly shadow the richer buckets.

4. **Difficulty heuristic is solid (0.97).** Keep it.

## Recommended fixes (next lift — CPU-only)

- **A. Tighten error detection.** Only treat `_is_error` as a failure
  when the marker is in a *tool result* of an **executable** tool
  (`terminal`/`bash`/`execute_code`) or an explicit `error` field — NOT
  in `read`/`read_file`/file-content outputs (which routinely contain
  the word "error"). This alone should cut most FPs and shrink the
  `debug` over-prediction.
- **B. Refine the bucket cascade.** Require `debug` *intent*
  (`fix`/`bug`/`traceback`/`why`) rather than mere `has_error AND
  shell`; emit `shell`/`refactor`/`docs` when those dominate.
- **C. Re-run the pipeline after A+B**: `analyze` -> `verify` ->
  `verify-exec` -> `mine-repairs` -> `bench-build`, so the balanced
  corpus, the failures set, and the 27 repair pairs reflect corrected
  labels.

The validation harness (`src/validate_classifier.py`) is committed and
reproducible; only the hand-labeling step is manual.

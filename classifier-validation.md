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

## Fixes A+B implemented (this turn, CPU-only)

- **A.** `extract_features` now skips `_is_error` on `read`-group
  tool outputs (file contents routinely contain the word "error").
  Explicit `error` fields still count.
- **B.** `classify_bucket` cascades edit/search/shell/data/docs
  *before* the `has_error -> debug` fallback, and `debug` only
  wins outright on debug *intent* — so an errored session that
  edits files is classified by its edit/file nature, not force-bucketed
  to `debug`.

Re-validated with the **same 30 hand-labels** (recomputed
predictions only):

| metric | before | after |
| --- | --- | --- |
| bucket accuracy | 0.50 | **0.60** |
| error precision | 0.53 | **0.57** (fp 7→6) |
| error recall | 1.00 | 1.00 (tp=8, fn=0) |
| difficulty accuracy | 0.97 | 0.97 |
| `failures.jsonl` count | 1338 | **1249** (−89 false positives) |
| `debug` sessions | inflated (dominant) | **645** (reasoning 1505 now largest) |
| debug precision | 0.29 | **0.50** |

Pipeline re-run (all CPU-only, writes only to staging `analysis/`):
`analyze` (2945 sessions → 80 tasks, 1249 failures) → `strata`
(refreshed balanced 10k) → `verify` (49/80 = 0.613, unchanged) →
`verify-exec` → `mine-repairs` (**27** repair pairs, unchanged) →
`bench-build` (**49** verifiable tasks, unchanged). The verify /
repair / benchmark counts are stable because they rest on *file-target*
signals, not the `debug` over-prediction — so the earlier
deliverables stay valid.

**Remaining (optional):** a few `shell`/`multi-file-refactor` sessions
are still misrouted (precision 0.50 / 0.33) and `docs`/`mixed`
are rarely emitted; tightening intent keywords or adding an
`automation`/`cron` bucket would help but the marginal value
is small now that the dominant `debug` inflation is fixed.

The validation harness (`src/validate_classifier.py`) is committed and
reproducible; only the hand-labeling step is manual.

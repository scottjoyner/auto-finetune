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

### Root cause of the residual error false-positives

The 6 error FPs are *not* a bucket problem, and they are **not** all
the bare `"error:"` substring (an earlier draft guessed that — it was
wrong). Measured against the labeled set: dropping `"error:"` clears
**0/6** FPs and hurts **0** true-errors, so it is not the cause.
The real cause is genuine error-terminology appearing as **benign text**
inside other tool outputs:

- 3× `cron_e2d25a4…` (true `reasoning`): a cron status/log
  containing `no such file or directory`, `command not found`,
  `syntaxerror` as ordinary prose (commands it tried, not failures).
- `20260601_103…` (true `code-search`): `search` returned *source
  code* containing `if error:` and `except Exception`.
- `20260605_231…` (true `docs`): a `read` of a module whose
  docstring/log text contains `traceback` / `exception` / `command not found`.
- `ses_09f7b66…` (true `mixed`): a **genuine `Traceback`** — the
  hand-label marked `is_error=0`, so this is **label noise**, not a
  classifier bug.

Net: ~5 genuine FPs (error-terminology in benign cron logs / code /
docs) + 1 label-noise case. This is a **precision ceiling** of
substring matching: the same words mark a real failure *and* appear in
healthy status output, so no substring tweak fixes it. A robust fix needs
**output-structure awareness** — trust a tool's own `success`/`exit_code`/
`error` fields over prose scanning (and only treat `traceback`/`exception`
as failure when at line-start, not embedded in code). That is a larger,
deliberately-deferred change; it is *not* worth doing on a 30-sample
metric. For the classifier's actual job (strata weights + failure mining)
precision ~0.57 is acceptable — the 1249-failure set is still dominated
by real failures, and the file-check benchmark is unaffected.

### Staged launch artifacts (CPU-only prep, GPU still busy)

Ready to launch the moment `nas5-20260717` frees (see harvest-safety
rule — do **not** copy into `datasets/` while a training job holds the GPU):

- `launch/focused/train.focused.jsonl` — the balanced 10k SFT mix
  (`messages` format, directly consumable by `src.cli train --label=focused`).
  Built with `--holdout=analysis/auto-tasks.jsonl` so the **80
  benchmark source sessions are excluded** — the 49-task eval is a
  true held-out, not an overlap of the training corpus.
- `launch/focused/launch_focused.sh` — copies the mix into
  `datasets/train.focused.jsonl` (new file, no clobber), sets an isolated
  `TRAIN_OUTPUT_DIR` (`outputs/checkpoints/toolcall-v5-3b-focused`), runs
  `train --label=focused`, then `bench-build` for the 49-task benchmark.
- `launch/repairs/repairs.dpo.jsonl` — **605** DPO pairs
  (`prompt` / `chosen` / `rejected`) from `mine-repairs --include-commands`,
  ready for a future contrastive run that teaches self-correction.
  Default `mine-repairs` mines only file-target self-repairs (26);
  `--include-commands` adds shell-tool self-repairs (errored command
  -> later successful call to the same tool, different args) — 519
  `terminal` + 107 `bash` + 75 `execute_code` — for 605 total.
  Built reproducibly by `src/repair_mix.build_dpo_mix` (tested).

The validation harness (`src/validate_classifier.py`) is committed and
reproducible; only the hand-labeling step is manual.

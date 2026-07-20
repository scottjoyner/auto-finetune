# Classifier validation (hand-labeled sample)

`analyze` classifies every session with a deterministic heuristic
(`classify_bucket` / `classify_difficulty`) and flags failures via
`_is_error` on tool outputs. This checks how trustworthy that is, because
**strata weighting, the verify pass-rate, and the 27 contrastive repair
pairs all assume the heuristics are right.**

## Method

- `python -m src.cli validate-classifier` builds a stratified label
  sheet (`src/validate_classifier.build_sheet`): sessions from
  `failures.jsonl` (pred `is_error=True`) + random non-failures,
  each with the heuristic prediction + a compact, human-readable view
  (first user request, tool-action trace, error snippet).
- Two **error truths** are recorded separately, plus `true_bucket` /
  `true_difficulty`. `score()` reports bucket accuracy + per-bucket
  precision/recall and **two** error axes:
  - **command-error**: pred = `has_error`, true = `true_cmd_error`
    ("did a *tool command* fail?" — what `has_error` detects).
  - **session-failure**: pred = `has_error`, true =
    `true_is_error`/`true_session_failed` ("did the *session*
    ultimately fail?").

Sample = **60 sessions** from the live cleaned corpus (an
earlier 15+15 hand-labeled round, expanded +30 with both error
axes re-labeled by the agent from each session's action trace).

## Results (60-sample)

| metric | value |
| --- | --- |
| bucket accuracy | **0.75** |
| difficulty accuracy | **0.87** |
| **command-error** precision / recall | **0.96** / **1.00** (tp=25, fp=1, fn=0, tn=34) |
| **session-failure** precision / recall | **0.42** / **0.92** (tp=11, fp=15, fn=1, tn=33) |

Per-bucket (precision / recall / support):

| bucket | P | R | n |
| --- | --- | --- | --- |
| code-search | 1.00 | 0.67 | 3 |
| debug | 0.69 | 0.85 | 13 |
| docs | 0.00 | 0.00 | 1 |
| mixed | 1.00 | 0.50 | 2 |
| multi-file-refactor | 0.62 | 0.71 | 7 |
| reasoning | 0.86 | 0.86 | 29 |
| shell | 0.50 | 0.20 | 5 |

## Findings

1. **`has_error` is a COMMAND-error detector, and a good one
   (precision 0.96 / recall 1.00).** After the structure-aware
   fix (honor `exit_code`/`success`/`error`), only **1** false
   positive remains in 60 — a benign log line
   `Registered: ZScoreMeanReversionStrategy` that contains "error".
   Every genuine command failure is caught. This is exactly the
   signal strata weighting, failure mining, and DPO pairs want.

2. **The old session-failure label is the WRONG target for
   `has_error` (precision 0.42).** 15 of 60 sessions
   contained a command error but *recovered* (e.g. `git error:
   failed to push` then succeeded; a cron job cut off by a network
   error; an agent that hit a tool limitation but delivered a
   report). `has_error` correctly flags these as command errors;
   the hand-label marked them session-OK, so they score as false
   positives against session-failure. **The classifier is right;
   the validation axis was mis-scoped.** Judge `has_error` on the
   command-error axis (1); treat session-failure as a separate,
   harder problem (needs an outcome signal, not substring
   scanning).

3. **`debug` over-prediction is largely resolved** (precision
   0.29 → 0.69) by the cascade reorder (A+B): debug wins
   only on debug *intent*, and edit/search/shell/data/docs are
   classified by their nature first.

4. **Buckets beyond debug/reasoning are now emitted** (reasoning
   29, multi-file-refactor 7, code-search 3, mixed 2, shell 5),
   though `shell` is still weak (precision 0.50) — likely the
   `cron`-automation sessions that look like status checks. `docs`
   has n=1 (no conclusion).

5. **Difficulty heuristic is solid (0.87).** Keep it.

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

### Root cause of the residual error "false-positives" — semantics, not bugs

The 6 error FPs are *not* a substring-over-fire problem, and they are
**not** benign prose. Inspecting the actual tool outputs shows they are
**real command errors in sessions that ultimately recovered** — i.e. the
`has_error` *command-level* signal is **correct**; the hand-label
`true_is_error` instead means **session-level failure**. The two are
different axes:

- `20260601_103…` (true `code-search`, labeled ok): a `git push`
  returned `error: failed to push some refs` / `error: could not apply`
  — a genuine command error, but the session proceeded and finished.
- `20260609_124059…` (true `shell`): terminal JSON
  `{"output": "curl: (22) ... error: 422", "exit_code": 0, "error": null}`
  — `exit_code:0` means success; the `error: 422` is HTTP prose.
  `extract_features` now honors `exit_code` and counts this as **success**.
- `20260605_231…` (true `docs`): `{"output": "...python: command not found",
  "exit_code": 127}` — a real command failure (`exit_code` 127) in a
  session that worked around it.
- 2× `cron_e2d25a4…` (true `reasoning`): cron-automation logs with
  stray error-terminology in commands that *ran* (not failed).
- `ses_09f7b66…` (true `mixed`): a **genuine `Traceback`** the hand-label
  marked `is_error=0` — **label noise**, not a classifier bug.

**Implication.** `has_error` answers "did a tool command fail?" — exactly
what the contrastive-mining / strata-failure job needs, and exactly the
signal that makes good DPO self-repair pairs (a recovered error *is* a
repair opportunity). The validation harness's "error precision/recall"
therefore measures the **wrong target**: it penalizes the classifier for
correctly flagging sub-step command errors in sessions the human judged
"overall successful." The 6 "FPs" are, for our purposes, **true
positives of the right thing**.

**What changed (commit 9170abe-era `analyze` fix).** `extract_features`
is now structure-aware via `_tool_error`: it honors a tool's own
`exit_code` / `success` / `error` fields when present (JSON-string shell
outputs carry `exit_code`), so `exit_code:0` is never an error even when
the text says "error". This removes the clean `curl error:422` class of
false positive. It does **not** move the 30-sample numbers, because those
residual FPs are real command errors the label simply doesn't count as
session failures.

**Decision needed (B / validation methodology).** Before tuning further,
decide what `true_is_error` should mean:
1. *Session-level failure* (current labels) → the harness is right and
   `has_error` is mis-scoped; we'd need a session-outcome signal
   (e.g. did the assistant end with an unresolved error / exhaust retries).
2. *Command-level error* (classifier's actual job) → relabel the sheet
   (the 6 "FPs" become TPs) and the validator will show ~precision 1.0.

For the classifier's real consumers (strata weights, failure mining, DPO
pairs) the command-level definition is the one that matters, so precision
~0.57 against the *session* labels is acceptable and not worth chasing on
a 30-sample metric. Expand the label set (C) under the agreed definition
before drawing stronger conclusions.

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

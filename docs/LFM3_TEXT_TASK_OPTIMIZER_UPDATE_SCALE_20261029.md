# LFM3-1.7B LoRA task-quality and update-scale research — October 29, 2026

**Status:** isolated CPU-only research. These procedures do not authorize production training, model routing, release, or checkpoint export.

## Task and scoring protocol

Pinned public official `LiquidAI/LFM3-1.7B` BF16 base SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. Frozen backbone, rank-2 Q/V LoRA (44 trainable matrices, 146,432 parameters), CPU-only, no network or GPU. Train on **24** procedurally generated arithmetic/parity/greater-than language examples per seed, with 9 distinct model-selection questions and 9 *unexamined* future questions per seed. No personal or private traces. This procedural two-choice exercise is not a standard benchmark and does not measure general reasoning or production usefulness.

The evaluation metric is **teacher-forced correct-answer log-probability exceeding one deliberately incorrect foil**, not free-generation exact-match. Record both total correct/9 and correct-minus-foil log-probability margins per seed, and real step gradient/update/time telemetry. The final nine-item split is generated and hashed but **not evaluated or tuned on**. No outcome of this study can license optimizer promotion.

## Executed equal nominal learning-rate control

Full comparison on x1-370, three paired seeds × two optimizers × 20 actual updates each = **120 update steps**, AdamW LR `1e-4` and **experimental factor-wise Muon** LR `1e-4`. Strict `src.lfm3_lora_text_task_review` returned `review_required`: baseline and final heldout selection accuracies were unchanged for all six arms:

| Seed | Before | After AdamW | After Muon |
| --- | --- | --- | --- |
| 1 | 5/9 | 5/9 | 5/9 |
| 2 | 7/9 | 7/9 | 7/9 |
| 3 | 6/9 | 6/9 | 6/9 |

However confidence margins moved differently. Those are *descriptive*, not a decisive quality outcome. No task-accuracy improvement was observed, no final test was opened, and no adapters were exported.

Across these six arms, mean parameter update norms were about `0.02419` for AdamW versus `0.01469` for Muon (about `1.65×`). Equal nominal optimizer LR is therefore not matched effective update scale. Both received the same max-gradient norm clipping of `1.0`.

## Prospective update-scale diagnostic — preregistered before execution

- Repeat identical frozen base, deterministic curriculum, three seeds and 20 steps, AdamW control LR `1e-4`, experimental factor-wise Muon LR **`1.6e-4`**. This is a coarse, preselected update-scale diagnostic based on **training-only mean parameter update norms**, not selected from task holdout performance.
- Record every actual step, raw pre-clipping gradient norm, effective parameter update norm, wall time, initial/final adapter fingerprints, data-partition hashes, paired selection score and task-family accuracies.
- Check if Muon and AdamW aggregate update scales converge (do **not** assume strict proportionality between LR and update norm), and inspect both effect size and errors by seed. The final partition must remain unexamined.
- Do not search for a 'winner' from two LR values or from 9 selection questions; all optimizer suggestions stay advisory, and significance/retention/generalization are not established.

Reproduce in an isolated x1-370 worktree, under CPU-only offline environment, with explicit `--research-only --allow-real-lora-training --allow-experimental-muon-lora` flags. No running LM Studio instance is modified. Raw evidence must pass `python -m src.lfm3_lora_text_task_review REPORT.json` before committing it.

## Requirements before genuine self-tuning research

The next meaningful loop requires genuine held-out public-language tasks with answer adjudication, task families diverse from training, multiple independent seeds and a frozen external final test. Research must separate factor-wise Muon from a theoretically appropriate low-rank Muon variant. Only a separately authorized isolated training controller can consider bounded changes after independent evaluation. Existing Xwing recovery-latch, training admission and GitHub coverage gates must remain active. A synthetic paired-foils score cannot demonstrate autonomous recursive improvement.

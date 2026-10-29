# Full BF16 LFM3 fabricated 8-case paired A/B — 2026-10-27

**Research-only, nonproduction.** Exact artifacts:
- `src/lfm3_fabricated_ab.py` real-model runner.
- `research/fixtures/lfm3_fabricated_eight_case_v1.json` eight fictional tasks with distinct session hashes, four dev and four heldout.
- `research/results/lfm3_fabricated_eight_case_ab_20261027.json` full first-run aggregate receipt without raw answers.
- `src/lfm3_ab_protocol.py` updated to compute *paired wins and losses against retrieval*, not just against no-memory baseline; `tests/test_lfm3_ab_protocol.py` explicitly tests a regression.

## Model and controls

Actual pinned `LiquidAI/LFM3-1.7B` BF16 weights, SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`, on xwing CPU inside a network-disabled bubblewrap namespace with only research/model mounts read-only. Isolated Transformers 5.21.0; temperature 0, fixed seed, `max_new_tokens=40`, official chat template with `enable_thinking=False` **for all conditions**. A matching synthetic SensorGraft Rust head-gate plan is supplied for the gated conditions. No weights changed; no source trace data, database access or live retrieval.

This is a *fabricated reference-text* experiment, not actual SensorGraft vector or graph retrieval. The test cannot establish performance on user traces.

## Exact answer totals from the first eight-case run

| Condition | Dev n=4 | Heldout n=4 | Total n=8 |
| --- | ---: | ---: | ---: |
| Baseline | 0 | 0 | 0 |
| Supplied fictional reference | 3 | 4 | 7 |
| Synthetic head gates only | 0 | 0 | 0 |
| Reference plus synthetic gates | 3 | 3 | 6 |

A separate repeat of the real-model experiment using the upgraded paired evaluator confirmed **heldout: 0 paired wins and 1 paired loss for reference+gates versus reference alone**. Do not interpret one synthetic heldout loss as a generalizable degradation estimate, but do not ignore it either.

The default `enable_thinking=True` template had earlier exhausted the 40-token budget with zero exact matches. This is why the thinking-mode flag is part of the controlled experiment and cannot be changed between A/B conditions.

## Interpretation and guardrails

**Reference text helped** the fictional tasks because the required facts were included directly in the user context. **Synthetic untrained head gates did not help** and showed a net exact-answer regression on this tiny test. No trained controller, semantic memory, statistical significance, or field retrieval is demonstrated. P50/p95 latency from only four observations per split is not a fleet performance SLA. Output JSON retains only per-split scores, timing and token counts, never full answers.

SensorGraft PR #7 now provides a 384-dimensional categorical *shadow* feature vector for `trace-memory.v1` with `may_generate_gate_plan=false` and `may_train_controller=false`. The Python hook rejects its schema. Do **not** pass shadow vectors through `--synthetic-only` and claim they are trusted memory activations.

## Next bar for an improved model stack

At least a materially larger, human-reviewed, permissioned held-out session set is needed to compare:
- no-memory and no-gate baseline;
- authenticated SensorGraft retrieved context;
- retrieved context + pre-approved trained controller;
- retrieved context + conventional trained LoRA.

Any gate candidate should be evaluated by **paired** wins/losses against retrieval-only, not only against an easy no-memory baseline. No promotion, fleet rollout, trace-training or Neo4j replacement before CI, independent source-provenance and rollback gates pass.

### CI status at this publication

Current hosted repository test run has **1,113 passed, 6 skipped**, no failed test cases, but fails the preserved **69% minimum coverage requirement** (actual 56.43%). Xwing training-admission workflow passes. This PR must remain draft.

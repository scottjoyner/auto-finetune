# Real LFM3 BF16 categorical-index prompt experiment: negative outcome

**Date:** 2026-10-27. **Auto-finetune PR #61 remains draft.**

## End-to-end components

- SensorGraft PR #7: `sg_lfm3_category_lab` uses a real exact angular index over a **fabricated** SensorGraft journal. It selected an older `error`/`tool` trace from 16 prior session events, not merely the newest record.
- `src/lfm3_category_bridge.py`: strict Rust/Python packet-binding validator, requiring both the original chronological packet and the ranked packet, exactly matching the session, revision, source SHA, and event objects.
- `research/fixtures/lfm3_category_source_packet_20261027.json` and `research/fixtures/lfm3_category_ranked_packet_20261027.json`: frozen actual Rust outputs for the same disposable journal.
- `src/lfm3_category_search_smoke.py`: BF16 pinned LFM3-1.7B on xwing CPU, `enable_thinking=False`, greedy 16-token limit, and three prompt conditions. Tests include two prompt-policy-isolation checks.
- `research/results/lfm3_category_bf16_xwing_20261027.json`: complete exact result receipt without raw generations. Checkpoint SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.

## Actual outcome

Question: *What exact status token is listed for category_match_1? If no such result was provided, respond unknown.*

| Condition | Expected | Correct? | Observed decode tokens |
|---|---|---|---:|
| No reference | unknown | Yes | 2 |
| Chronological metadata (no category_match_1) | unknown | No | 16 (budget exhausted) |
| Categorical ranked metadata | error | **No** (model returned unknown) | 2 |

The exact ranking did its job: a semantically uninteresting but category-specific previous `error`/`tool` event was selected from the actual Rust journal. **The model did not correctly quote that unverified ranked event in this prompt.** It also did not produce an exact answer within the 16-token cap for the chronological condition. This failure is preserved, not hidden or re-scored.

This experiment is deliberately too small to estimate helpfulness, adversarial resistance, or memory gains. It contains no real trace data, no learned controller, no vector embeddings from language, no authenticated source generation, no model-weight changes, and no ROCm/GGUF performance test. The "synthetic lab" source label is not independent provenance attestation.

## Next steps

Improve literal metadata extraction without rewarding false source-truth claims, expand to reviewed multi-session cases and longer controlled decode budgets, then separately measure retrieval quality and answer accuracy. Test learned head gates only after the retrieval-only baseline and synthetic/head-gate regressions are understood.

Keep the PR draft; do not lower its coverage threshold, merge to production or migrate Neo4j based on this experiment.

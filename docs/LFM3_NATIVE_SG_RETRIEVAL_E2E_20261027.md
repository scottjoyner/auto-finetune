# LFM3 BF16 × native SensorGraft session retrieval: first end-to-end lab

**2026-10-27; auto-finetune draft PR #61.** Uses the real pinned `LiquidAI/LFM3-1.7B` BF16 model, a **fabricated** SensorGraft Rust journal, and an offline read-only research sandbox. It is not a production trace pipeline or quality benchmark.

## Interfaces and verified artifacts

- SensorGraft PR #7: `sg_lfm3_retrieve_lab --synthetic-lab-only` reads a disposable Rust journal and emits `sg-lfm3-retrieval-metadata.v1`.
- `src/lfm3_retrieval_bridge.py`: strict Python parser with schema/size/duplicate/finite/categorical/order checks, independent recomputation of the Rust domain-separated item fingerprint, and explicit denial of source-trust elevation. Renders at most eight categorical event rows without exposing event IDs, session hashes or source digests to the model. It **does not authenticate** the source.
- `src/lfm3_native_retrieval_smoke.py`: real BF16 prompt runner, network-disabled sandbox admission, pinned snapshot verification, `transformers==5.21.0`, `enable_thinking=False`, greedy 16-token max. No gates, LoRA or model updates.
- `research/fixtures/lfm3_native_session_ok_packet_20261027.json` and `research/fixtures/lfm3_native_session_error_packet_20261027.json`: exact generated Rust packet fixtures; their category records are artificial.
- `research/results/lfm3_native_retrieval_2x2_xwing_20261027.json`: complete paired receipt, hashes and timings only, without raw model outputs.

## Two distinct questions

`extract`: quote the *listed* `previous_event_1` status, regardless of the journal's authority. Gold comes from the **fabricated fixture** only.

`authority`: report independently verified true status, or `unknown` if the only input is this legacy-unfenced journal. Gold is `unknown`; the test rewards epistemic caution rather than unverified source promotion.

| Fabricated session | Task | No reference | Rust packet reference |
|---|---|---|---|
| Latest event `ok` | Extract listed status | Incorrect (`unknown`) | Correct (`ok`) |
| Latest event `error` | Extract listed status | Incorrect (`unknown`) | Incorrect (`unknown`) |
| Latest event `ok` | Independently verified truth | Correct (`unknown`) | Correct (`unknown`) |
| Latest event `error` | Independently verified truth | Correct (`unknown`) | Correct (`unknown`) |

The first ordinary "what is the status?" experiment was ambiguous and cannot distinguish source extraction from verified truth. The explicit split above makes that limitation visible.

**Observed:** Rust session retrieval, packet schema parity, strict Python custody flags and real BF16 LFM3 context consumption work for two tiny fabricated sessions. Context changed only one extracted answer successfully. The model abstained on the unverified-source truth question as desired. There is no evidence of semantic vector retrieval, broader task quality improvement, trained head gates, adversarial prompt-injection resistance or independently authenticated source custody.

## Reproduction and restrictions

Use the SensorGraft synthetic-journal commands in `docs/LFM3_NATIVE_SESSION_RETRIEVAL_20261027.md`, freeze the two returned packet JSON files, then inside an **unprivileged, network-disabled, read-only** sandbox exposing only the pinned model snapshot and research files:

```bash
python3 -m src.lfm3_native_retrieval_smoke \
  --offline-only --ack-sandbox-research --task-mode extract \
  --snapshot /approved/cache/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd \
  --packet /approved/research/fabricated-packet.json
```

Never run that command against a production journal, host-mounted `/nas`, credentials or a writable vendor-code directory. Application-level environment flags alone are **not** a sandbox. The pinned checkpoint SHA-256 is `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.

## Next steps

Increase independently reviewed sessions and test extraction against genuine source evidence **only after authorization, separate custody, and privacy review**. Require paired controlled comparisons of retrieval-only, baseline, gates, retrieval+gates and a conventional LoRA. A separate future authority witness must protect monotonic source generations; local journal checksums or self-asserted flags are insufficient.

Keep both PRs drafts. No Neo4j replacement, real trace content ingestion or training has been authorized by these lab observations.

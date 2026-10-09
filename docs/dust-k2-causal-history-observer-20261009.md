# Dust/K2 R2.1 — additive causal history in real PRE witness

**2026-10-09 | Research-only | Stacked on [round-two PR #25](https://github.com/scottjoyner/auto-finetune/pull/25), source-cohort #24, direction witness #23 and classifier #22**

## Accepted implementation slice

The read-only `LocalProbeWitness` and `k2_real_direction_probe` now accept an **explicitly opt-in** causal-history v2 mode (`--collect-causal-history-v2`). In every PRE record, the producer writes the eight-component `auto-finetune.dust-k2-direction-history.v2` vector *before* running that direction's ± forward. These features are calculated exclusively from **earlier completed** plus/minus and clean scored losses; an entire 4-direction batch shares the same prior-history snapshot. History state resets per keyed masked-prompt source group.

The old `features_pre_probe` eight geometric/proxy scalars remain unchanged, and **derived `auto-finetune.dust-predictive-probe.v1` labels remain byte-schema compatible**, so existing strict derived-label and receiver PRE/POST checking can still run. This is an **additive PRE event field**, not a new learned classifier or training route.

The v2 `CausalProbeHistory` now rejects skipped/replayed PRE indices, partial label commits and commit batches without exactly matching prior previews. Prior completed losses are committed into the history state only **after every POST/derived record in that batch has been fsynced**.

An independent, pure-Python `k2_causal_evidence_verify.py` reconstructs the history sequence using only the chain-linked PRE/POST events. It rejects stale, future-derived, malformed or extra/sensitive event fields; checks strict source, label and model hashes; verifies PRE/POST ordering and the actual derived-label fields. The verifier reports `PASS_PRODUCER_CAUSAL_REPLAY_ONLY`—it does **not** pretend to have independent receipt signing-key custody or re-compute the true K2 numerical forward losses.

## Validated behavior

**x1 isolated CPU environment:** 4/4 strict history tests, 6/6 real-estimator synthetic witness parity/receipt-join tests (including new history-enabled parity), 3/3 standalone causal replay/tampering tests passed.

**Xwing real-model bounded smoke:** Existing pinned local pretrained K2-Horizon-0.9B ROCm weights; previously approved source at sample index 1 under the *exact* existing 128-token cohort preflight; K=8 shared orthogonal antithetic directions, sigma=.25, direction batch 4, seed 42. The model loaded from the local checkpoint without a download.

**Observed outcome:** 8/8 complete real directional plus/minus loss observations, **2/8** beneficial plus perturbations, frozen K2 base/LoRA parameters, zero backward/optimizer steps. The standalone causal replay verified all eight PRE/POST and derived labels and showed history fields were causally reconstructible from earlier POST losses. The original v1 label schema was preserved.

Local SSD evidence on Xwing under `/media/scott/data/finetune-staging/research-witness-20261009/`:
- `historyv2-smoke8.events.jsonl`: SHA256 `1899f4baaaac8cbca05ee31e4a4cfb84de020285217499c0421125331b83e899`
- `historyv2-smoke8.derived.jsonl`: SHA256 `591d505696ce883b408f34c186c06bc15256648c0160e1d1cd944ae198660bfd`
- `historyv2-smoke8.summary.json`: SHA256 `a56ab076c46f48690194ced915fe0359ee30c338b0fcfb057ddadc14234e7ec6`

The verifier's read-only acceptance fields are `history_pre_only_recomputed=true`, `producer_event_chain_verified=true`, `original_label_schema_preserved=true`, `receiver_receipt_count_not_independently_signed_here=0`, `receiver_signing_key_isolation_accepted=false`, `classifier_training_authorized=false`.

## Replay

```bash
cd /media/scott/data/git/wt-dust-k2-causal-witness-20261009
python -m experiments.dust.k2_causal_evidence_verify --verify-only \
  --events /media/scott/data/finetune-staging/research-witness-20261009/historyv2-smoke8.events.jsonl \
  --derived /media/scott/data/finetune-staging/research-witness-20261009/historyv2-smoke8.derived.jsonl \
  --summary /media/scott/data/finetune-staging/research-witness-20261009/historyv2-smoke8.summary.json
```

Only the explicit `--collect-causal-history-v2` mode adds history to future experiments. The default path still creates original PRE/derived evidence. No old evidence file is overwritten, replayed or promoted.

## Remaining gates, in sequence

1. **G0 restricted receiver UID/key HOLD:** The previous x1 receiver's HMAC key is still readable from Xwing through the shared `scott` SSH principal. This smoke deliberately used **no** receiver receipt to avoid pretending independent key custody was corrected. A future receiver service must be restricted, rotated and tested from Xwing's producer identity; existing receipts cannot be upgraded retroactively.
2. **G1 source adequacy HOLD:** Lexical cross-corpus scanning from PR #25 has not run successfully against the additional local corpora. Semantic/near-paraphrase independence and ≥64 train, 16 validation and ≥32 untouched heldout source groups remain unproven. Eight directions from one source are **not** eight independent training examples.
3. **G2 real combined feature schema NOT_RUN:** History v2 provides **eight episode-history scalars**, **plus** the existing eight geometric/candidate-specific proxy scalars. They have not been combined into a frozen, independently audited real-classifier input vector. The toy synthetic ensemble weights **must not** be transferred. Historical features may be identical within each 4-direction batch, so they alone cannot rank directions.
4. **G3 held-out classifier and quality NOT_RUN:** Fit a new source-disjoint classifier only after G0–G2 pass; compare with random, simple/curvature-aware momentum and matched-cost K2 CE and bootstrap intervals. The original synthetic classifier **failed** independent confirmatory superiority; no predictor selection or optimizer distribution changes authorized.

**Disposition:** Additive PRE history and complete producer-local causal replay **PASS**. Independent receiver security, classifier training, source adequacy, production optimizer and deployment **HOLD/DENY**.

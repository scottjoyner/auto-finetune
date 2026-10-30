# LFM3 optimizer methods — third forward-only Dust arm registered (2026-10-30)

**Research-only; SHADOW/INSPECT, not an executable training API.** This branch provides a concrete, common evidence-backed interface for AdamW, experimental factor-wise Muon, and predictive Dust with its *genuinely trainable external 195-weight LFM3-attention-conditioned cross-attention head*. It does not loosen Xwing admission/recovery holds, repository coverage, production routing or checkpoint promotion rules.

## Why this integration

The original Dust/K2 backprop-free experiments (draft PRs #13 and #22–24), the bounded forward-only LoRA-B research arm (draft #86), causal activation attention (#87), online head-weight calibration (#88), real matched-scope AdamW/Muon/Dust experiments (#89), prequential zero-baseline test (#90), uncertainty abstention (#91), derivative-free 195-weight neural head (#92) and new task-family transfer (#93) already established substantial experimental code.

Previously, however, those methods lived in separate research scripts with no **single reproducible description of their optimization type, trainable parameter scope, verified trial comparisons, head admission status, and permissible action**. This branch implements that interface as `src/lfm3_three_method_shadow_registry.py`, with an actual read-only trained-head bridge `src/lfm3_predictive_dust_shadow_bridge.py`.

## Three methods, one source-validated research interface

| Method | Model update mechanism in earlier isolated trial | Backprop | Common scope | New method-admission status |
| --- | --- | --- | --- | --- |
| AdamW | LoRA-B gradient / AdamW step | Yes | 2,048 final V-projection LoRA-B scalars | Historical isolated research; no live permission |
| Experimental Muon | LoRA-B gradient / factor-wise Muon step | Yes | Same 2,048 | Historical isolated research; no live permission |
| **Predictive Dust** | Bounded antithetic forward-only directions / train-loss acceptance | **No** | Same 2,048 | **Blocked** pending credible positive head forecast study |

**Important distinction:** The historical bounded Dust model-update runner `src/lfm3_dust_predictive_probe.py` is not yet wired to *actuate* the new 195-weight neural head. The trained head is connected only via a pure, strictly read-only **shadow candidate-ranking bridge**. This is intentional: across the new 3-seed extraction-to-uppercase family-transfer study (#93), its prospective MSE was **1.116 / 1.171 / 1.126 times worse than predicting zero gain**. The head's learned coefficients are real, but it is **not admitted** for model-update control. No benchmark result warrants automatic optimizer switching.

The single interface recomputes rather than trusts the historical matched adapter study. It loads exactly **12 pinned local JSON evidence files** (three seeds × AdamW/Muon/Dust, three seeds × trainable neural-head transfer), hashes each, calls strict lower-level structural reviewers, and enforces matching task manifests, initial LoRA-B hashes, exact adapter names and parameter counts, and the unchanged compute-inequality flag. If any input is missing, malformed, tampered, symlinked or outside the fixed evidence inventory, it blocks.

Only the research inspection action is supported. `--intent=train`, `--intent=promote`, `--method=auto` and all unrecognized method names block. There is **no hidden override**. Even AdamW and Muon are not executable through this registry; their existing explicit research-only runners are separate.

## Actual trained-head bridge

`src/lfm3_predictive_dust_shadow_bridge.py` independently rebuilds all 195 Q/K/null/mixer parameters of each already-trained neural head from its **12 public extraction-task training episodes**, applies the last legal training-only update, validates the frozen learned-weight SHA-256 against the original LFM3 evidence, then forecasts and ranks eight signed LoRA directions for a previously unseen public uppercase-task prompt. It verifies the ranking's predicted gains match the values recorded **before any transfer-probe outcomes were measured**. It does not use transfer-family measured gains to update the head, nor load or write a model.

Actual shadow replay rankings, highest prediction first:

| Seed | Direction indices, sorted by trained head | Applied to LFM3 weights? |
| --- | --- | --- |
| 10 | 3, 0, 4, 7, 6, 5, 1, 2 | **No** |
| 11 | 3, 1, 5, 7, 6, 4, 0, 2 | **No** |
| 12 | 0, 4, 7, 3, 6, 2, 5, 1 | **No** |

The differing rankings demonstrate actual head inference from different learned weights and LFM3 activation data, not predictor success. Every replay returned `predictive_head_shadow_abstention`, `selected_for_live_probe: null`, zero additional LFM3 forwards, and zero model-weight updates.

## Run the review

```bash
python3 -m src.lfm3_three_method_shadow_registry \
  --research-only --method all --intent inspect
python3 -m src.lfm3_three_method_shadow_registry \
  --research-only --method predictive_dust --intent inspect
python3 -m pytest -o addopts= -q \
  tests/test_lfm3_three_method_shadow_registry.py \
  tests/test_lfm3_predictive_dust_shadow_bridge.py
```

The inspector outputs a JSON method catalog with all evidence SHA-256 checksums, per-seed matched results, parameter/compute-equality status, actual neural-head transfer ratios, and explicit `model_training_permitted: false`, `optimizer_switch_permitted: false` and `weight_promotion_permitted: false`. File output is exclusive create (`open('x')`). No model checkpoint, GPU, Hugging Face network, privileged credentials, inference service or training process is required for inspection.

The local 32-test suite passed, including adversarial tests for false method admission, malformed provenance, overridden training intent, head prediction tampering, evaluation-feedback leakage and model-update promotion. Hosted CPU-only CI verifies the same. The original source reports are **not cryptographically independently attested** and the common interface does not fix the historical Dust signing-custody issue; it validates only the checked-in evidence structure/math.

## Next test, not yet approved

Before enabling learned Dust to *select real LoRA update directions* or switch the optimizer, preregister a broader independent public-text training/selection/final-test protocol with task-family-disjoint labels, prediction uncertainty calibration, gain-per-model-forward/walltime/energy, exact A/B adapter parameter matching and rollback. Keep a cheap zero baseline, random and curvature-aware momentum controls; compare AdamW/Muon only under matched resources. Future policy eligibility must pass new independent evidence **and** independent runtime custody and human review. The head should have permission to abstain (or be disabled) if its prediction gains are statistically indistinguishable from noise. No auto-adjustment should be self-authorizing.

**Merge and release gates unchanged:** research PR remains draft; full repository coverage target **69%** is not met (latest predecessor PR #93 head **58.03%**, 1,404 pass / 3 skip), and Xwing, independent code review, safe deployment, weight provenance and other recovery gates remain in force.

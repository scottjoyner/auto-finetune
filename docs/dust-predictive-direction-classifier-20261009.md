# Predictive multi-direction stochastic fine-tuning — classifier research slice

**2026-10-09 | Draft research only | Parent: Dust/K2 PR #13, issue #1 | No K2 trainer or scheduler integration**

## Decision

Build a **low-cost auxiliary utility classifier** to predict which perturbation directions might reduce local loss, in **shadow mode**. This is not a replacement for the fixed highest-fidelity shared orthogonal K=1024 / sigma=0.25 / D=4 estimator. The classifier cannot affect direction sampling, gradient estimates, LoRA updates, deployment or release.

No prior direction-level supervised K2 trace is available in this branch. Aggregated gradient-cosine and heldout CE evidence from Dust/K2 PR #13 is **insufficient** to train a legitimate direction classifier: its labels must come from full, completed, independently witnessed per-direction probes. Until then, all learned weights here are from **synthetic quadratic data only**.

## Working model

- Three independently bootstrapped **regularized logistic heads**, each trained by CPU stochastic-gradient-style optimization (100 epochs) on the **same fixed training-only cohort** of synthetic task episodes. Every head bootstraps full episodes rather than mixing directions from the same episode across partitions.
- Input vector consists only of values available *before scoring a proposed direction*: recent and older historical gradient/probe projections, momentum projection, estimated local curvature, step-size terms, and disagreement. No current candidate's measured plus/minus loss is a feature.
- Target label is **beneficial direction** for one synthetic quadratic candidate: `actual_gain = -[eta*g·d + (eta²/2)*dᵀHd]` > 0. This is not actual K2 next-token cross entropy or model-quality gain.
- Episode splits are predefined: 64 train ×16 directions, 16 validation ×16, 16 untouched test ×16. Validation only selects an F1 threshold from a frozen grid. Test is used for reporting, not training/parameter selection.
- Report top-four precision and gain against **random**, **momentum**, and **curvature-aware momentum**. Calculate paired 1000-resample bootstrap intervals over **episodes**, never over candidate rows.

## First x1-370 completed experiment

Python 3.12.3, standard library only, no pretrained weights, GPU, model/scheduler mutations or hosted inference. Deterministic result JSON at:

`/home/scott/git/dust-predictive-classifier-synthetic-v2-20261009.json`

SHA256: `718a00adb7dc5927b48e96d3bac9eee156b4ba2ed6eae4c8b5d553a7506f477a`.

| Strategy | Test precision@4 | Test mean true gain@4 |
| --- | ---: | ---: |
| Logistic ensemble | **0.921875** | **0.39881145** |
| Momentum | 0.906250 | 0.39656276 |
| Curvature-aware momentum | 0.906250 | 0.38726287 |
| Random | 0.406250 | -0.10186693 |

**Small-sample guard:** Ensemble minus curvature-aware momentum precision = **+0.015625** (paired episode bootstrap 95% CI **[-0.03125, +0.0625]**); mean true gain = **+0.0115486** (CI **[-0.0149102, +0.0429988]**). Both intervals include zero. Claims of a robust advantage, generalization or heldout model improvement are **NOT SUPPORTED**. The classifier is not admitted as a sampling policy.

The synthetic ensemble produces a model-coefficient SHA256 in the report, but no model is promoted, downloaded, or published as production weights. Its ensemble uncertainty estimate is also recorded.

## Forward-probe data contract, for later real labels only

New module `predictive_probe_contract.py` defines a **schema-only validation gate**. It accepts exactly: `schema`, opaque keyed-HMAC episode digest, pinned model-revision SHA256, candidate index, eight **pre-probe** features, sigma, clean loss, + perturbation loss, - perturbation loss, and an explicit time-order attestation flag. Raw prompts, token IDs, direction vectors, images, activations, credentials, and arbitrary JSON fields are refused. Invalid/larger-than-bounded records, duplicate candidates, mixed model versions, and non-finite values fail closed.

All candidates from the same episode are assigned to the same hash-derived partition. The plus-direction label is `loss_clean - loss_plus > 0`; the antithetic minus loss is retained *only for analysis*, never used as a pre-probe feature.

**Critical limitation:** a self-asserted JSON boolean does **not** cryptographically prove that features existed before candidate loss was known, that the HMAC is keyed/custodially protected, or that source prompts are held out from K2 model fine-tuning. Consequently `validate_jsonl` returns `classification_training_authorized=false`, even on valid records. A separate producer/receiver signed-timestamp and data-custody witness is required before any K2 collection or classifier training on derived real labels.

## Direct reproduction

```bash
cd /home/scott/git/wt-dust-predictive-classifier-20261009
python3 -m unittest discover -s tests -p 'test_dust_predictive*.py' -v
python3 -m experiments.dust.predictive_direction_classifier \
  --synthetic-only --output /path/to/NEW-synthetic-result.json
```

No real-data training CLI is provided; `--synthetic-only` is mandatory and output creation uses exclusive `open("x")`.

## Next independent gates

1. **Read-only witness on per-direction K2 probes:** producer-owned, HMAC-grouped source IDs; pre-probe features frozen before plus/minus scoring; no prompts/IDs in returned summaries; full K directions still evaluated in shadow so labels are not selectively missing. A positive time-order boolean alone fails this gate.
2. **Data adequacy:** ≥3 independent source groups per label class and at least 32 heldout source episodes before fitting real probe records; check class balance, duplicate prompt/near-duplicate screens, and cross-model version shifts. Split by source/prompt and run, never candidate row.
3. **Training:** fit identical pretrained-free logistic/ensemble baselines to real derived training records; tune hyperparameters/threshold on validation only. Compare top-K precision/gain, Brier/calibration, wall time and variance against momentum, curvature and random; report per-episode bootstrap CI. Do not claim 'fine-tuned K2' simply because a separate classifier was trained.
4. **Sampling-bias safety:** any later learned direction selection changes the perturbation distribution; a valid importance-weighting / unbiased estimator or explicit biased-policy experiment and matched-cost control are needed. **Never silently reduce K** based on this predictor.
5. **Quality gate:** heldout K2 CE and tool-call quality across independently pinned 8/16/32-update schedules (Dust PR #13 follow-up), matched ordinary backprop and shared-orthogonal controls. Direction classification accuracy alone cannot authorize live training.
6. **LittleBit interaction:** keep scales-only QAT as a separate hypothesis with its own real-weight and storage gates. A classifier may eventually prioritize forward-only scales perturbations, but the two research programs are not yet integrated.

**Current gate: SHADOW SYNTHETIC PASS; REAL-LABEL TRAINING NOT_RUN; LIVE TRAINER/SCHEDULER/GPU/NAS/DEPLOYMENT DENY.**

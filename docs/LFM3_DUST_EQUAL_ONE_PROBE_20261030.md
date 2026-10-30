# Real LFM3 predictive Dust attention head vs random: equal one-probe pilot
*October 30, 2026 — x1-370 CPU-only, research-only, draft*

**Decision: HOLD / NOT QUALIFIED for autonomous model updates.** The genuinely trained external 195-parameter LFM3 prediction-attention head can rank forward-only model-adapter directions using prior measured perturbation gains without backpropagation. This trial reduces an important prior confound: earlier neural-v-greedy experiments measured eight candidate perturbations to pick the greedy winner. Here **neural and random each physically receive exactly one measured training candidate forward** at the same 0.05 LoRA-B trust radius. It still does NOT establish equal *total compute*, because neural needs an extra LFM3 prompt-context forward and historical head fitting/inference.

## Frozen protocol and source details

- Pre-registered fresh procedural seeds **25,26,27**, paired respectively with the earlier trained-head sources **10,11,12**. Pinned official BF16 LiquidAI/LFM3-1.7B, isolated offline CPU inference on x1-370; no public network, production services or user content.
- Head: previously trained **195 weight, two-head external attention predictor**, rebuilt from its prior extraction-training records using bounded derivative-free head-weight proposals. It reads one **actual LFM3 prompt-only attention-context forward**, then ranks 8 unmeasured directions (four ± pairs). No head updates from the new diagnostic examples. Fixed deterministic random chooses from the same eight *without using the head* or measured gains.
- Scope: final two attention V-projection **2,048 LoRA-B scalar adapter parameters**, all other model weights frozen. One new public first-word extraction training prompt per seed. Four distinct extraction and two uppercase transfer diagnostic examples per seed; no prompt IDs overlap the source training set. Before any model-perturbation outcome, record both selectors' chosen directions. **Each policy measures just one training perturbation**; if that candidate improves the training prompt, it is applied **temporarily** for six separate diagnostic forwards. If it fails, the policy abstains and still scores six diagnostic forwards at the clean adapter to preserve evaluation forward budgets. The initial baseline consists of one clean train forward and six diagnostic forwards, shared by both policies.
- Actual measured per seed: **1 context forward, 21 completion-loss model forwards**: 7 shared baseline (1+6), 1 random probe + 6 random diagnostics, 1 neural probe + 6 neural diagnostics. Across seeds: **63 real completion-loss model forwards, 3 neural-context forwards** and 3 frozen head reconstructions. No 8-probe greedy oracle and no model backprop; **zero retained adapter changes or exports**. Each ephemeral adapter perturbation restored by exact tensor copy and starting/final SHA-256 match. Source execution receipts are unsigned local JSON, not independent hardware attestations.
- Policy *probe + diagnostic* runtime and attention-context/forecast/head-reconstruction runtime separately logged. The benchmark executes the random policy first and neural policy second; cache/warm-order effects are NOT counterbalanced. There is no measured power/energy budget and no gradient AdamW/Muon run. Consequently, the comparison is **one-probe-model-budget matched, not total walltime, energy or three-optimizer compute matched**.

## Real, not simulated, diagnostic results

Change in six-prompt **diagnostic completion loss** relative to the same initial adapter. **Negative is better; positive is worse; 0 means no temporary step.**

| New seed | Random one-probe | Neural-head one-probe | Neural signed direction | Neural train gain |
| --- | ---: | ---: | --- | ---: |
| 25 | 0 (abstain) | 0 (abstain) | 2 | −0.0000358 |
| 26 | 0 (abstain) | **−0.008756** | 6 | +0.020698 |
| 27 | 0 (abstain) | **+0.004735** | 2 | +0.000268 |

On seed 25, both selectors happened to choose **the same** signed direction (index 2); both separately measured negative training gain and abstained. On seed 26, neural chose a direction that improved four same-family diagnostics by **−0.013358**, but **worsened two separate uppercase diagnostic losses by +0.000448**. The weighted overall diagnostic difference improved **−0.008756**. On seed 27, neural chose a positive-training-gain step that improved same-family diagnostic loss **−0.001786** but **degraded uppercase transfer loss +0.017776**, leading to **+0.004735 total diagnostic degradation**. The random control abstained on all three new seeds, making its diagnostic change zero in this narrow cohort. This is **not** evidence that all random one-probe directions are poor.

Equal-seed neural mean diagnostic difference is approximately **−0.00134** vs zero for random, but it comprises just one benefit, one harmful update and one abstention; **do not promote the model/head based on this mean**. Even successful overall improvement can mask a damaging transfer-family regression.

Measured per-policy probe+diagnostic elapsed seconds (random/neural): seed25 **2.266 / 2.536**, seed26 **2.511 / 2.411**, seed27 **2.799 / 2.853**. Extra neural CPU head reconstruction ~**0.16s**, frozen LFM3 prompt context ~**0.6s**, eight neural attention scores ~**0.004s** per trial, not counted as random policy cost. CPU timing has ordinary benchmark noise, includes different acceptance paths and is not compute- or energy-normalized proof.

## New files, independent math checks and preservation rules

- `src/lfm3_dust_equal_probe_benchmark.py`: explicit opt-in `--research-only --allow-frozen-head-equal-probe-counterfactual`, only seeds 25–27, rejects output path already existing **before** expensive model load, and writes one fixed-scope experiment source receipt per seed.
- `src/lfm3_dust_equal_probe_review.py`: deterministically rebuilds historic 195-weight neural head; replays all 8 pre-probe forecasts, deterministic random choice, strict one-probe budgets, training gains, diagnostic arithmetic and LoRA restored SHA receipts. It validates math and internal provenance only; it does NOT cryptographically attest the original model runtime.
- `src/lfm3_dust_equal_probe_admission.py`: strict cohort admission checks that flags **seed27 diagnostic/uppercase regressions**, lack of per-seed gains, and neural extra compute; always refuses model training and promotion.
- Raw real data `research/evidence/lfm3-dust-equal-probe-20261030/seed{25,26,27}-matched-one-probe.json`, plus new focused/adversarial tests and dedicated hosted CPU-only CI.
- The shared research-only three-method `src/lfm3_three_method_shadow_registry.py` now validates **27 complete research source receipts** (historical AdamW/Muon/Dust, trained-head task transfer, task-shift, true one-step diagnostic, multiprompt, variance proxy, and equal-one-probe experiment). Source manipulation, extra probes and misleading promotion claims block. **No training-intent execution path exists in this registry**.

## Next scientifically useful direction

For an adaptive LFM3 prediction-head method to become a viable third optimizer, it must maximize positive diagnostic improvement *without regression across task families*, not only train-prompt gain. Plan a larger prospective trial across genuinely distinct public task families, evaluate correct answers and retention, use two-sided uncertainty and drift detection calibrated exclusively on prior training examples, and compare AdamW/Muon/forward-only Dust under **total** compute, token, walltime and power budgets. Record independently signed execution evidence. Leave existing locked final test/production gates untouched and do not tune this model using the evaluated seed25–27 diagnostics.

**Status:** three seeds and only one actual update per policy; promising isolated seed26, harmful seed27. Method admission `hold_not_qualified`, draft/unmerged, no persistent LoRA weights saved, no automatic optimizer switching. Unchanged **69% repository-wide coverage gate** and Xwing/release recovery controls apply.

# Real pretrained LFM3-1.7B: predictive Dust neural-head one-step quality pilot
*Isolated CPU-only, in-memory, research-only protocol — October 30, 2026*

**Decision: HOLD / do not use neural head to select production fine-tuning updates.** This follow-up tests whether a proposed LoRA-B direction produces any improvement on *separate diagnostic examples*. Previously, the 195-weight forward-only trained neural attention head learned from historical extraction training gains but failed prequential zero-baseline and cross-instruction-family forecasting controls. The latest attention task-shift safeguard also vetoes optimizer actuation. The explicit offline counterfactual described here **does not override that admission gate**.

## Fixed study, committed before real-model execution

- Three registered new public procedural seeds **16, 17 and 18**, each paired with a fully audited frozen 195-parameter trained head from seed 10, 11 or 12 respectively. Exact pinned official BF16 LiquidAI/LFM3-1.7B loaded with offline CPU-only preflight on x1-370; 2,048 scalar last-two-attention-`v_proj` LoRA-B scope. All other pretrained/LoRA weights frozen and every temporary tensor perturbation reverted by **exact copy**. Adapter initial/final hashes match for every run; no model or adapter export, checkpoint, gradients, backprop, torch optimizer object or production service change.
- Fresh independent-from-selection public text task examples (distinct prompt IDs from the original head training set and one another): **one first-word extraction training prompt per seed**, four new same-family extraction diagnostic examples and two uppercase-rewriting transfer diagnostic examples. Diagnostic outcomes are **never used** to train the attention head, choose a direction, set a trust radius or calibrate the task-shift threshold. These are **procedural synthetic tests**, not final model-quality or real-world generalization evidence.
- Before any new candidate gain is measured, acquire a prompt-only LFM3 activation summary and freeze all eight signed direction predictions from the historical trained 195-weight neural head. Generate four normalized random directions and their ± signs in the previous fixed eight-dimensional LoRA-B subspace.
- Measure **all eight signed candidates** at a fixed adapter radius of **0.05** (upper trust limit 0.15). For each policy, choose exactly one direction or abstain: (a) no update, (b) deterministic random one candidate, (c) neural-head top-ranked one candidate, (d) empirically best training-gain direction from eight forwards. A selected direction is only temporarily applied if it produced strictly positive measured training gain; otherwise it abstains.
- For every policy, take the same fresh pretrained adapter as the reference, temporarily apply its selected 0.05-radius LoRA-B direction **only inside the isolated CPU process**, and score exactly six diagnostic completion losses. Restore the adapter by exact copy and assert the initial SHA-256 again. All policy branches share the same initial model, prompts, parameter scope, update radius and evaluation metrics; **candidate-probe budgets are unequal** (random=1, neural=1, greedy=8, zero=0). Since every candidate and every policy were evaluated in this study, actual process compute is **not** equal to these prospective one-policy budgets.
- Explicit flags `--research-only --allow-isolated-unqualified-head-counterfactual` are mandatory. The head is deliberately **unqualified** to actuate live updates; we evaluated it as a one-step scientific counterfactual in-memory. No training controller is enabled by this script or its output.

## Actual measured results

Change in mean completion loss on six *unseen diagnostic prompts* relative to the unchanged adapter, after **one temporary** signed LoRA-B update. **Negative is better. Positive means worse.** These are loss measurements, not generation-based exact-answer accuracy.

| Method | Seed 16 | Seed 17 | Seed 18 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Unchanged model | 0 | 0 | 0 | 0 |
| Random, one candidate forward | 0 | 0 | 0 | 0 |
| 195-weight neural-head top candidate, one forward | 0 | 0 | **+0.017225** | **+0.005742** |
| Empirical best of eight measured training candidates | **+0.013316** | 0 | **+0.010125** | **+0.007814** |

- Seed 16: neural head's selected direction **hurt the training prompt** (training gain −0.003131), and so **no step** was made. Greedy measured a **+0.035524 training gain**, temporarily applied that direction, but **worsened separate diagnostic completion loss by +0.013316**. The same-family extraction diagnostic was worse by +0.021666, while uppercase loss improved by −0.003385.
- Seed 17: all eight proposed directions had negative training gain. Neural, random and empirical greedy **all abstained**, so no policy changed any diagnostic score.
- Seed 18: neural picked a direction with **+0.001427 training gain**, but its one temporary step **worsened overall diagnostic loss by +0.017225** (same-family +0.029597, uppercase −0.007517). Greedy picked +0.029342 training gain, but **worsened overall diagnostic loss by +0.010125** (same-family +0.015040, uppercase +0.000295). Random again abstained.
- Across three seeds, 24 actual signed perturbation-model forwards, 3 clean train-prompt forwards, **90 diagnostic loss forwards** (baseline + four policy evaluations per seed), 3 LFM3 context forwards, a total of **117 completion-loss model forwards**. The head itself performs no *new* parameter updates here; its 195 trainable parameters come from the verified prior source. **Zero retained model updates and zero reported backward calls**.

The concrete failure mode is **per-prompt overfitting**: selecting by positive gain on a single training example does not ensure improvement on another prompt even from the same family. This pilot is too small for robust estimates, and repeated diagnostic evaluation must not be used to optimize its hyperparameters retroactively. The neural attention mechanism can learn and rank candidates, but it currently does not reliably guide useful actual model fine-tuning.

## Replay, admission, and common optimizer registry

`src/lfm3_dust_sandbox_update_quality.py` runs the real single-step study. `src/lfm3_dust_sandbox_update_review.py` reconstructs the historical neural head's trained Q/K weights, verifies the frozen preprobe LFM3 context, all eight predicted signed directions, the fixed deterministic random and greedy choices, positive-train-gain accept/reject decisions, diagnostic data manifest, measured quality deltas, adapter hashes and model-forward counts. **The independent Python reviewer verifies math and data contracts, not independent hardware custody or the authenticity of reported model-loss values.** All local runs are unsigned source-authored telemetry.

Source evidence is committed under `research/evidence/lfm3-dust-one-step-quality-20261030/seed{16,17,18}-one-step-diagnostic.json`. `src/lfm3_dust_sandbox_update_admission.py` rejects training-policy promotion unless every seed exhibits improvement in both same-family and transfer diagnostic quality as well as an advantage over the random one-probe control. Here **`hold_not_qualified`**. Even a future passing result would authorize a *separate confirmatory study only*, never production training.

The shared `src/lfm3_three_method_shadow_registry.py` now requires **18 structural research receipts**: nine earlier matched-parameter AdamW/Muon/Dust records; three trained-head cross-family transfer records; three prompt-only out-of-distribution guard records; and three actual diagnostic quality-update records. It refuses a new optimizer or model-training request while any input is missing, forged or disallowed. The parameter-matched AdamW and Muon comparisons remain from earlier isolated training; **this is NOT a new compute-matched AdamW/Muon/Dust training benchmark**.

## Research recommendations

1. **Use more than one train prompt per update decision** and check a lightweight *separate* training-side guard sample before considering a parameter change; the current single-prompt reward is unstable.
2. Separate head Q/K learning from adapter controller admission. Do not optimize to the existing six diagnostic examples. Study conservative predicted-gain lower bounds, task-conditional calibration and measured gain-per-total-forward-equivalent on *fresh* preregistered task families.
3. For genuine three-method science, run exactly matched 2,048-parameter AdamW/Muon/Dust update pilots with equivalent total compute, walltime/energy and independent diagnostic correctness/retention metrics.
4. Preserve original untouchable final-test, independent execution attestation, Xwing training admission, protected merge and the **69% repository-wide test coverage floor**. No autonomous recursive self-improvement, weight export, production service modification or unreviewed merge is enabled by this PR.

**Status: research-only, draft, no persistent model update.**

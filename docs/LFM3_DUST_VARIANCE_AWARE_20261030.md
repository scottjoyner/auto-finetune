# LFM3 forward-only Dust: training-gain variance pilot — October 30, 2026

**Research decision: HOLD / no automatic fine-tuning.** This experiment adds a third, deliberately fixed acceptance rule for a genuinely trained 195-parameter external LFM3 activation-conditioned attention predictor. It remains the **third experimental optimizer strategy**, alongside AdamW and Muon, but the new evidence does not qualify it for model-weight promotion, live optimizer switching or recursive autonomous fine-tuning.

## Experimental question and frozen protocol

The earlier three-prompt Dust research (#99) showed that requiring three positive training-prompt gains is insufficient: it can reject beneficial adapter perturbations and accept harmful ones. This preregistered pilot replaces the crude positivity-only check with a **variance-aware training-gain rule**. For exactly three independent training gains `g=[g1,g2,g3]`, allow a temporary LoRA perturbation only if **all three gains are positive AND mean(g) − sample_std(g) > 0**. This is a heuristic one-standard-deviation margin, **not a statistically valid lower confidence bound**; three observations cannot establish robust probabilities.

New seeds **22, 23, 24** (not selected based on their outcomes) reuse the previously frozen 195-weight neural heads from source seeds 10, 11, 12. On each, the official pinned BF16 LFM3-1.7B loads offline CPU-only on x1-370. A new public procedural task supplies three distinct extraction training prompts (first/last/first) and six fully separate diagnostics (four extraction and two uppercase). The head's prompt-only 4D LFM3 activation and eight signed LoRA candidate forecasts are frozen **before** the first training-gain measurement. A deterministic random selector, trained neural-head top-1 selector, and an eight-probe first-prompt greedy control use the exact same ± perturbation candidates. Compare three acceptance rules on each selected direction: first-gain-positive, all-three-gains-positive, and the new variance margin. Only after all train-only acceptance decisions are fixed are candidate diagnostic changes evaluated, including for candidates later vetoed. All policies share an identical frozen model initial state and 2,048 last-two-attention V-projection LoRA-B scalars at 0.05 perturbation radius, and every perturbation is restored with an exact SHA-256 match. No gradients, model save, retained adapter updates, private transcripts, locked final-test examples or production process changes.

## Real experimental results

Mean diagnostic completion-loss difference after a *temporary* one-step LoRA perturbation. Negative = improved, positive = degraded; zero = no update.

| Seed | Neural, single-prompt | Neural, all 3 positive | Neural, variance | Greedy, single-prompt | Greedy, variance | Random, variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | **−0.006502** | 0 | 0 | **−0.003284** | 0 | 0 |
| 23 | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | **−0.017049** | **−0.017049** | **−0.017049** | −0.008815 | −0.008815 | −0.008815 |

The neural head did select directions that improved *separate* synthetic diagnostic completion loss on **seed 22 and seed 24** under the one-prompt research rule, and on seed 24 under the stricter variance rule. This is an encouraging *preliminary* signal, not a statistically reliable model fine-tuning capability or win versus no update/random on a large independent task distribution. The extra margin also **vetoed seed 22's useful neural direction** (and the greedy useful direction); on seed 23 everyone abstained. The variance rule did not demonstrate superiority to a simpler zero/no-update rule across all seeds. Even where it accepted a direction, there was no saved model update.

Specific train-only confirmations: seed 22 neural candidate had positive first-prompt gain and gain **+0.000711** on the second example but **−0.001085** on the third; the strict rule vetoed it, although it improved later diagnostics. Seed 24 neural candidate passed both confirmation prompts with gains **+0.003955**, **+0.000881** (in addition to its positive first gain), and improved diagnostics by **−0.017049**. This selection may be task/procedure specific; do not fit another threshold to these now-observed diagnostic outcomes.

Total real pinned LFM3 work: **83 completion-loss forwards** across seeds (33/17/33), including 24 signed candidate first-prompt forwards, plus **3 actual prompt-only context forwards**. These are experiment-execution totals, not demonstrated savings of an operational policy; random, neural and greedy candidate-probe budgets differ, and no walltime/energy-matched AdamW/Muon benchmark was run. Head weights were reconstructed from earlier audited training receipts; **zero head training updates and zero model backprop calls in this new pilot**, with exactly matching starting/final adapter hashes.

## Source custody, tests and hard controls

- `src/lfm3_dust_variance_quality.py`: preregistered seed22–24 counterfactual runner, explicitly requires `--research-only --allow-unqualified-variance-counterfactual`; comparison arms run entirely in memory and restore all LoRA-B perturbations.
- `src/lfm3_dust_variance_review.py`: deterministic replay reconstructs the **historical learned 195-weight attention head** and current preprobe forecasts, signed proposal directions, training-only positive/variance decisions, every model-forward count, independent diagnostic loss arithmetic and restoring model SHA. Source-run JSON is **unsigned**; mathematical replay is not independent runtime attestation.
- `src/lfm3_dust_variance_admission.py`: observes a beneficial head-selected update vetoed by the variance rule; returns `hold_not_qualified` and never offers automatic model-update authorization, irrespective of outcomes.
- `research/evidence/lfm3-dust-variance-20261030/seed{22,23,24}-variance-gated-quality.json`: all three real pinned-checkpoint source receipts, no checkpoint/adapter exports.
- `src/lfm3_three_method_shadow_registry.py`: common research-only AdamW/Muon/Dust evidence interface now requires **24 source receipts**, including the prior matched-scope, trained-head transfer, task-shift, one-step quality, three-prompt and new variance studies. Missing, fabricated or changed evaluations block all method registration/actuation.

**Scientific conclusion:** A derivative-free trained LFM3 prediction head can sometimes select beneficial temporary directions. An acceptance rule based on three training gains and their sample variance is **not yet superior**: it can lose useful updates, and reliable generalization, real task correctness, compute-normalized gain and safe training effects remain unproven. The next experiment must use substantially more new tasks, independently sealed evaluation metrics, and compute-normalized actual AdamW/Muon/Dust outcomes; do not optimize against the already-observed seed22–24 diagnostics.

Existing **69% full-repository coverage minimum**, Xwing admission and recovery gates, protected human review, and no-deployment/no-weight-promotion requirements remain unchanged. This branch is **draft, stacked and unmerged**.

# LFM3 predictive Dust: three-prompt forward-only verification experiment

**Research only — October 30, 2026. Decision: HOLD.** This branch advances the third, backprop-free optimizer arm alongside AdamW and Muon. It tests whether two *separate training-side verification prompts* help prevent a direction selected by the pretrained LFM3 attention head from hurting unseen diagnostic examples. Results **falsify** the hypothesis that requiring positive gains on all three training prompts reliably ensures diagnostic improvement.

## Previously established baseline

Draft #92 trained 195 parameters of an external two-head LFM3 activation-conditioned Q/K attention network using forward-only ± head-coordinate proposals and previously measured training gain labels. Its actual weights update without autograd or backpropagation. However, subsequent independent transfer studies (#93) showed its gain forecasts lose against trivially predicting zero, an OOD guard (#97) abstains under unfamiliar activations, and true ephemeral LoRA-B updates (#98) could improve immediate training loss while **worsening independent diagnostic loss**. The trained head therefore remains unqualified to actuate autonomous model updates.

## New registered experiment and actual model execution

**Precommitted before measurements:** new seeds **19, 20, 21**, paired with the independently audited frozen attention heads from previous seeds **10, 11, 12**. The backbone is the pinned official BF16 `LiquidAI/LFM3-1.7B`, isolated, HF offline and CPU only on x1-370. Exact 2,048 scalar last-two-attention-V-projector LoRA-B scope, perturbation radius **0.05**, strict trust maximum 0.15, all other weights frozen. No model checkpoint export, gradients, optimizer object, selection/final-test access or production changes. Every temporary perturbation restored with exact-copy `finally`, with starting/ending adapter SHA-256 equal.

Per new seed, generate **three distinct extraction training examples**—first-word, last-word, first-word—and **six distinct diagnostic examples**: four new extraction examples and two uppercase-rewriting examples. All are different public procedural prompt keys, and diagnostic outcomes are never used for candidate selection, head training, the three-prompt accept/reject rule or the OOD threshold.

Before any measurement, compute the real LFM3 4-scalar prompt-only attention activation and the **eight frozen predicted gains** for four LoRA-B directions and both ± signs. Evaluate all eight directions on the first training example. Three candidate selection policies share these measurements: deterministic random choice, top neural-head predicted gain, and empirically best-of-eight measured first-prompt gain. For every distinct selected candidate with positive first-prompt gain, obtain two **additional train-only** gain measurements on the second and third training examples. Then compute the same six-prompt diagnostic score for that candidate even if the three-prompt rule rejects it, to expose false rejections. No duplicate diagnostics are re-evaluated for the single vs multi rules; all comparison arms reuse the exact same candidate outcomes.

Compare **one prompt** (apply temporary step if first prompt gain >0) against **three prompts** (require first and both confirmations >0). Candidate updates remain temporary *counterfactuals*, not live training: the original admission gate already failed and is never overridden. All diagnostic comparisons begin from the same exact initial model. Measurement counts include forward passes for diagnostic baselines, all eight first-prompt directions, clean training losses and positive-candidate verifications. This experiment is **not** a compute-matched comparison to gradient AdamW or Muon, nor does it establish prospective cost savings in an operational one-policy workflow.

## Actual results: confirmation did not fix quality

Change in average completion loss on six new diagnostic examples (negative better, positive worse):

| Seed | Neural one-prompt | Neural three-prompt | Greedy one-prompt | Greedy three-prompt |
| --- | ---: | ---: | ---: | ---: |
| 19 | 0 | 0 | 0 | 0 |
| 20 | 0 | 0 | **−0.015108** | **0 (vetoed)** |
| 21 | **+0.012751** | **+0.012751** | **+0.012751** | **+0.012751** |

The random policy always abstained because its first-prompt choice had negative gain. Seed 19: all three selectors abstained. Seed 20: greedy selected a positive-gain candidate on the first prompt, but its two independent training verification gains were negative, so three-prompt gating rejected it; diagnostic evaluation showed that the rejected update **would have improved** diagnostic loss by 0.015108. Seed 21: neural and greedy chose the same direction, with first-prompt gain **positive** and both further training verification gains **positive**; nevertheless, its temporary adapter update **increased diagnostic loss** by 0.012751.

The hypothesis “two more positive train-side gains ensure diagnostic improvement” is **rejected on these examples**. This is too small a set of synthetic tasks for estimates of false acceptance/rejection rates, but it provides concrete counterexamples to the simplistic guard. It also demonstrates why optimizing the acceptance rule against already observed diagnostic examples would leak evaluation outcomes into training policy. We have deliberately **not** retuned thresholds using these results.

**Measured source execution:** 17 / 25 / 25 completion-loss model forwards on seeds 19 / 20 / 21 = **67 real LFM3 loss forwards**, plus 3 prompt-only attention context forwards. All 24 first-prompt signed directions measured. Extra confirmation forwards: 0 / 2 / 2; candidate diagnostic forwards for positive-first proposals: 0 / 6 / 6. None retained an adapter change. No new head training was performed; the historical head was reconstructed and frozen from authenticated-by-structure local training receipts.

## Verification, gates and next experimental question

- `src/lfm3_dust_multiprompt_quality.py`: registered, fixed-seed real LFM3 one-step experiment with three train-only prompts, separate diagnostics, exact adapter rollback and required `--research-only --allow-unqualified-multiprompt-counterfactual`.
- `src/lfm3_dust_multiprompt_review.py`: checks original head source SHA and historical Q/K training replay, eight preprobe neural forecasts and deterministic directions, train-only confirmation signs, policy choices, separate diagnostic score arithmetic, model-forward accounting and unqualified policy status. **This is mathematical evidence verification, not independently signed execution custody.**
- `src/lfm3_dust_multiprompt_admission.py`: requires consistent benefit and no harmful acceptance or beneficial veto across seeds even to nominate a *separate* prospective study. Result **`hold_not_qualified`**. It authorizes neither model updates nor optimizer switching under any outcome.
- Three new source reports in `research/evidence/lfm3-dust-multiprompt-20261030/seed{19,20,21}-three-train-prompt-quality.json`; all are source-authored local JSON, not independent cryptographic attestation. The common `src/lfm3_three_method_shadow_registry.py` now reads and validates **21** historical/actual research receipts and refuses any missing or altered status; dedicated and inherited CI exercise this gate.
- Earlier AdamW/Muon matched-adapter trials still had unequal backprop and forward compute budgets. No new baseline-training performance data is claimed here.

**Next experiment worth preregistering:** a distribution of many fresh train-side prompts and independent diagnostics per seed; learn a task-conditioned latent gain/uncertainty model using only historical training receipts, then score **accuracy, precision/recall of beneficial updates, retained-task correctness, and walltime/energy per useful update**. Compare to gradient AdamW, factor-wise Muon and random/greedy forward-only controls with matched total compute. Keep head and parameter-update admission separate, use independent execution signing, and do not reuse the present six-prompt diagnostic examples for policy optimization.

All work remains draft, unmerged and offline; the existing **69% repository-wide coverage gate**, Xwing recovery controls and independent model-release review are unchanged.

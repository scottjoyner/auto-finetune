# LFM3 backprop-free predictive Dust: online single-gain attention-head training
*October 30, 2026 — actual pinned model runs on x1-370, research only*

**Result: online learning of the actual 195-parameter Q/K head works, but its model-update decisions did not improve.** The third experimental optimizer strategy remains predictive Dust (alongside AdamW and Muon). This research directly addresses a bottleneck in earlier work: the external LFM3 attention head could be optimized from historical **eight-gain batches**, but an operationally economical one-probe policy yields only **one gain per episode**. We now update the trained Q/K parameters online using a single *previous* train-only probe gain, with no gradient/backpropagation. The pretrained LFM3 model weights never persistently change.

## Implementation and causal update

The new `OnlineSingleFeedbackHead` starts from the previously verified 195-weight, two-head Q/K prediction network (source seeds 10–12). On fresh episode (t), it first trains on **exactly the previous episode's one measured neural-selected training gain**, while computing the previous prediction against the **memory snapshot from before that gain was observed**. It tries two fixed-seed ± antithetic head-parameter proposal pairs, evaluates previous-episode squared gain-forecast error by small head-only forwards, and accepts only strictly improving proposals within the existing 0.6 parameter-L2 trust radius. The new weights and history are then used for **current episode candidate forecasting**, before any current model outcome is measured. This is *real Q/K parameter adaptation*, not post hoc scalar score scaling.

The previous training gain comes only from one real LFM3 perturbed train-prompt completion-loss forward. The head is never fed current/future gain labels, independent diagnostic losses, locked evaluation partitions or private user traces. No autograd or backward calls. Every head-training step is replayable and checked by weight SHA-256. Head-only objective forwards are counted separately from all actual LFM3 model forwards.

## Preregistered actual pretrained LFM3 pilot

Three new public-text procedural seeds **28, 29, 30**, each with **three new chronological episodes 12, 13, 14**, with unique train and diagnostic prompts. Source heads paired with prior trained seeds 10, 11, 12; official pinned BF16 `LiquidAI/LFM3-1.7B`, offline CPU on x1-370. Each episode has one new extraction train prompt, four independent extraction diagnostic prompts and two uppercase-transfer prompts, with no prompt-ID reuse across a cohort.

Compare exactly three policies: **random** one signed candidate, **frozen 195-weight attention head** one candidate, **online Q/K-adapting head** one candidate. All policy selections are fixed before *any* current model training-gain measurements. Each policy uses the same eight deterministic ± direction choices, identical 2,048-scalar final-two-`v_proj` LoRA-B parameter scope, perturbation radius 0.05, **one real measured train-model perturbation**, and exactly **six independent diagnostic model forwards**. The model starts from the exact same adapter each arm/episode, is temporarily perturbed only when required, and is restored by exact tensor copy and SHA check after all arm evaluations. Only the adaptive head learns from its **own selected one** preceding training gain; the frozen head and random control never update.

Per seed: **3 prompt-context model forwards + 84 completion-loss model forwards** (three episodes × [one shared clean train and six baseline diagnostics + three policy arms × one train-model probe and six diagnostic forwards]). Total new LFM3 inference: **9 context forwards + 252 loss forwards**, zero model backprop, zero saved model checkpoints, zero retained model-adapter changes. Each real run was source-authored in local unsigned JSON, not independently cryptographically attested. Trial budgets are matched for measured model train perturbations across policies; **total compute is NOT matched** because Q/K head inference and training, context passes and execution order differ. Source Q/K initial heads themselves required historical probe data to train, which is not amortized in this small experiment.

## Verified observed results

| Seed | Online Q/K accepted head-only updates | Head-only training objectives | Online model-direction choices differing from frozen |
| --- | ---: | ---: | ---: |
| 28 | 2 | 7 | **0/3** |
| 29 | 2 | 7 | **0/3** |
| 30 | 2 | 4 | **0/3** |
| **Total** | **6** | **18** | **0/9** |

The head's 195-weight SHA-256 **changed** in all three runs. Yet every learned-head signed direction remained identical to its original frozen counterpart, even after adapting to the preceding probe. Thus each pair produced **the same per-episode diagnostic completion-loss changes**. Aggregate per-episode mean diagnostic-loss difference: **+0.000390** frozen **and** online (slightly worse than no-update baseline), while random mean was **+0.002310**; these are tiny synthetic cohorts and cannot establish statistical superiority. The head did not deliver incremental benefit for its additional inference.

Illustrative individual episodes: seed 28 episode 12 neural step **−0.007225** (improved) versus random **+0.012227** (harm), but seed 28 episode 13 neural **+0.004301** (harm). Seed 30 episode 12 both neural variants **+0.009497** (harm); episode 14 both neural variants **−0.003065** (improvement). The learned head is able to change numerical forecasts and weights, but this pilot found **no changed top-ranked direction**, and therefore no improvement in actual update decisions.

## Evidence gates and limitations

- `src/lfm3_dust_online_single_feedback.py`: derivative-free 195-scalar Q/K update from one preceding measured gain. Exact chronology, historic prior-memory snapshot, 0.6 L2 trust region, fixed-seed two ± proposal pairs; no future label or model-backprop access.
- `src/lfm3_dust_online_one_feedback_trial.py`: real, explicitly opt-in, offline three-episode x1-370 model runner; identical model probe budgets per random/frozen/adaptive policy and exact LoRA-B restoration; early refusal to overwrite existing source receipts.
- `src/lfm3_dust_online_one_feedback_review.py`: independently replays prior Q/K source head training, every new head-only proposal acceptance and weight digest, all current preprobe neural forecasts, fixed-seed directions, policy choices and diagnostic arithmetic. A mathematical reviewer, **not** independently authenticated LFM3 hardware execution.
- `src/lfm3_dust_online_one_feedback_admission.py`: fail-closed 3-seed evidence gate explicitly separates **successful head learning** from **model-quality improvement**. Reports `hold_not_qualified`; no optimizer switching, head auto-selection for production, model weight promotion or trust-based admission.
- `research/evidence/lfm3-dust-online-one-feedback-20261030/seed{28,29,30}-three-episodes.json`: all source-authored real-model data, with per-episode weight hashes, feedback history, 8 forecasts, 3 candidate probes and independent diagnostic losses.
- Shared read-only **AdamW/Muon/predictive-Dust registry now requires 30 research receipts** across matched-parameter gradient/forward experiments, trained-head transfer, OOD checks, actual diagnostic update pilots, single/multiprompt/variance guards, matched one-probe baselines and this true online one-feedback Q/K trial. All statuses must reconcile; any source alteration or promotion attempt blocks.

**Scientific decision:** A trainable attention prediction head now genuinely adapts at a one-probe model budget without backpropagation. *That alone is insufficient.* The next independent experiment must demonstrate that online updates actually change selected directions and improve retained-task correctness across new instruction families. Compare complete gradient AdamW/Muon and forward-only Dust at genuinely matched **total** compute/energy, with prospective preregistration, hidden final evaluation and independent source attestation. Do not retro-tune these weights on already-observed diagnostics.

**Release boundary unchanged:** draft/unmerged, full repository >=69% coverage gate, Xwing and human independent review, no production or persistent model training change. No unsupported claim of recursive self-improvement.

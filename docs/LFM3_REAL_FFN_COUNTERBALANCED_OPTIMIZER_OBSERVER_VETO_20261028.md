# Real LFM3 optimizer telemetry observer: label-reversal veto

**Date:** 2026-10-28. **Node:** xwing, offline research. **Status:** draft, non-executing.

Official BF16 LFM3-1.7B, pinned checkpoint SHA256 135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330, independently re-read prior real final-FFN LoRA 12-trial AdamW/Muon learning-rate telemetry. These experiments trained only 38,912 temporary rank-4 FFN parameters; the original 1,728,801,536-parameter pretrained backbone remained frozen. The training/dev prompts are tiny fabricated examples, not independently curated evaluation data.

The numeric evaluator verified that constant LR [0.004,0.004,0.004] outperformed a first-step [0.002,0.004,0.004] schedule in all three paired seeds for both optimizers. In the first observer test, the better option was labeled A, and LFM3 chose A in all four formats.

The counterbalance trial (source 20c62b744cb7a56ca29a4ebca2f39ae34aaf7075) reversed ONLY the labels on the same independently validated telemetry. B was then correct in all four prompts, yet the real original model still chose A for both AdamW and Muon in both formats. A/B next-token logits: AdamW JSON 20.5/13.5, text 21.875/16.5; Muon JSON 20.5/13.4375, text 21.625/17.125. Thus the overall score was **4/8**, with all eight choices A: evidence of a label/position preference in this limited setup, not reliable numerical optimization advice.

Evidence receipt: research/results/lfm3_real_ffn_label_reversal_four_failures_xwing_20261028.json.

New independent last-mile gate src/lfm3_ffn_optimizer_advisor_admission.py joins the fixed actual-model training receipt, prior numeric verification, original four votes and reversed four votes. It returns label_position_bias_detected=true and reviewable_hypothesis=null for the actual results. Nothing can automatically train, reconfigure native Muon/AdamW, write model weights or promote models based on those logits. Its tests deny forged source flags and contradictory logits.

SensorGraft draft PR #7 now imports ONLY small metadata and builds a validated 58-node/57-edge research graph with four rejected reversed votes and an external bias veto. A fresh isolated scratch journal replayed at revision 2. No NAS, Neo4j or serving stack was modified.

Blockers: source authentication/signatures are absent, existing parity heldout is spent, general task transfer is unproven, Python CI still has 58.21% coverage versus mandatory 69%, and SensorGraft parallel lock issue #8 plus source custody issue #9 remain open. Both PRs remain draft. This falsifies an overly optimistic observer claim; it does not demonstrate recursive self-improvement.

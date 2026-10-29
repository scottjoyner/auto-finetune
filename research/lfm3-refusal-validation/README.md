# LFM3-1.7B Refusal and Adversarial Validation — research prototype

**Status: research-only, NOT a validated safety mechanism or an unrestricted public model release.**

This is a standalone safety-focused companion to auto-finetune and sensorgraft.
The experiment studies how LFM3-1.7B's conv, sliding-attention, and full-attention
layers encode refusal-related signals. It compares post-trained and (later) base
checkpoints, reversibly perturbs activations, and tests benign utility, false
refusals, malicious-action resistance, and simulated authorization.

## Assets on x1-370
- Original source weights: /home/scott/lfm3-research/model-metadata
- Original Q4 benchmark: /home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf
- Python: /home/scott/lfm3-research/train-venv/bin/python
- Nothing in this repository changes or deletes the originals.

## Quick start
Run in the project directory:
  /home/scott/lfm3-research/train-venv/bin/python -m pytest -q tests
  /home/scott/lfm3-research/train-venv/bin/python -m src.refusal_probe --mode probe --max-pairs 2
  /home/scott/lfm3-research/train-venv/bin/python -m src.refusal_probe --mode compare --max-pairs 2 --layer 10 --alpha 0.25

The probe writes a private activation-direction safetensors file to the ignored
results directory; compare tests frozen-weight generation using reversible hooks.
Do not publish the vector, raw harmful responses, or derived weights without
separate safety and license reviews.

## Research methodology and limits
1. Freeze model SHA, tokenizer/template, seed, precision, and matched dataset.
2. Capture last-position hidden vectors from each decoder block with hooks.
3. Estimate per-layer harmful-minus-benign contrastive directions; do not assume
   these are causal refusal circuits without intervention and holdout validation.
4. Compare baseline and reversible projections; include sham controls and
   independent held-out cases. Current eight pairs are smoke tests ONLY.
5. Score false refusals, legitimate helpfulness, refusal and safe-redirection
   behavior on adversarial prompts, and mocked unauthorized tool attempts.
6. Require independent manual annotations for actionable harmful content;
   keyword refusal matches are not a sufficient safety verdict.
7. Inference is local and tool execution is disabled. Simulated actions never
   become real commands. No automatic deployment or model promotion.
8. Track regressions by category and publication gates in paper/main.tex.

## Critical limitations
- Correlation is not causation: steering can shift model style without changing
  effective resistance to malicious actions.
- Eight paired probes and a lexical refusal proxy cannot support broad claims.
- Base and post-trained variants must be evaluated separately; Q4 and BF16
  results must not be pooled.
- Work remains unpublished until enough independent human-graded holdout results
  exist, reproducibility is verified, and responsible-release review is complete.
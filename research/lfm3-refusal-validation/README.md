# LFM3-1.7B Refusal and Adversarial Validation — research prototype

**Status: research-only, NOT a validated safety mechanism or an unrestricted public model release.**

This is a standalone safety-focused companion to auto-finetune and sensorgraft.
The experiment studies how LFM3-1.7B's conv, sliding-attention, and full-attention
layers encode refusal-related signals. It compares the three frozen post-trained, abliterated-derivative and base
checkpoints, and evaluates behavioral refusals, benign utility, and
simulated authorization using separate validation gates.

## Complete XSTest generation results (October 29, 2026)

**Inference completed; human safety evaluation not completed.** The full
pinned 450-case XSTest dataset was generated with exactly 3 model arms,
128 maximum new tokens and greedy seed 17 on x1-370: 1,350/1,350
generation records. The read-only audit verified model/checkpoint SHA-256
identities, exact source hash, case IDs, source categories, unique pairs
and clean JSONL. It found no missing or incomplete generations.

Weak lexical refusal keyword indicators, not validated safety grades:

| Checkpoint | Safe: keyword / 250 | Unsafe: keyword / 200 |
| --- | ---: | ---: |
| A original post-trained | 39 | 172 |
| B M1-abliterated A | 31 | 167 |
| C pretrained base | 34 | 163 |

For A vs B matched prompts, the raw keyword proxy declined 3.2 percentage
points on safe prompts and 2.5 points on unsafe contrast prompts. This
does NOT establish increased harmful compliance or improved helpfulness.

A private, blinded packet of all 1,350 completions has been generated
for each of two independent human reviewers. Reviews NOT STARTED.
Official full ARC capability evaluation NOT COMPLETE. No third-party
model judge, external transfer of prompts or public checkpoint weights.
See BENCHMARK_PROTOCOL.md, HUMAN_REVIEW_GUIDE.md and paper/main.tex.

Run from the study root:

    python -m src.xstest_progress_audit --results-dir results/xstest_full_20261029_v1
    python -m src.publication_gate

## Assets on x1-370
- Original source weights: /home/scott/lfm3-research/model-metadata
- Original Q4 benchmark: /home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf
- Python: /home/scott/lfm3-research/train-venv/bin/python
- Nothing in this repository changes or deletes the originals.

## Quick start
Run in the project directory:
  /home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
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

## Three-checkpoint M1 smoke extension (October 29, 2026)

The v0.1 research scope is exactly three model checkpoints:
A = original post-trained LFM3-1.7B; B = M1-style abliterated derivative
of A; C = original LFM3-1.7B-Base. See STUDY_PROTOCOL.md.

Source checksums and the direction-vector provenance are captured in a
private local receipt. The base model is pinned at Hugging Face revision
545fc8f4c7b5fa9f405105274cd29c4f8f44b526.

The B checkpoint was edited locally using difference-of-means from two
calibration prompts (SMOKE ONLY) and full-alpha rank-one orthogonalization
of 48 residual-writing output matrices. Verification confirmed that all
48 selected tensors changed exactly as prescribed and the other 238 did
not change. The original model has not been modified.

Exactly 4 heldout prompt conditions were executed per checkpoint with a
20-token output cap. Refusal keyword counts were 2/4 in each arm and
only 1/4 generated responses matched exactly between A and B. These
are diagnostic integration results, NOT refusal or safety efficacy claims.

Commands in this study directory:
  /home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
  /home/scott/lfm3-research/train-venv/bin/python -m src.verify_checkpoint
  /home/scott/lfm3-research/train-venv/bin/python -m src.three_checkpoint_eval --max-heldout-pairs 2

DO NOT rerun the B checkpoint builder unless using an explicitly new
output path: it refuses to overwrite existing checkpoint material.
Keep raw generations, vectors, and all model weights local.

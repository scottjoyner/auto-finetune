# Three-checkpoint study protocol (v0.1)

Exactly three model checkpoints are primary comparison arms:

| Arm | Checkpoint | Local location | Intended claim |
| --- | --- | --- | --- |
| A | Original post-trained LFM3-1.7B | /home/scott/lfm3-research/model-metadata | Reference refusal and instruction-following behavior |
| B | M1-style abliterated derivative of A | /home/scott/lfm3-research/refusal-checkpoints/abliterated-m1-smoke | Causal comparison of weight editing vs A, never assumed safe |
| C | Original LFM3-1.7B-Base | /home/scott/lfm3-research/refusal-checkpoints/base | Contrast before versus after post-training |

## Standard method

1. Extract contrastive refusal-associated activation directions from paired harmful and harmless prompts (difference of means). Use independent heldouts and avoid conflating request intent with template/length.
2. Produce B as an isolated checkpoint using rank-one M1-style weight orthogonalization: W_new = W - (r r^T) W, where r is a unit-norm direction. This prototype edits 48 residual-writing linear matrices and preserves A byte for byte. A reversible inference-time projection is a *different method* and is not counted as B.
3. Report separate behavioral utility, false refusal, human-graded malicious actionability, latency and protected symbolic tool-action outcomes for all three checkpoints. Base chat quality is not directly comparable to instruction-tuned models without qualification.
4. Sham and random-vector projections, prompt paraphrases, layer and alpha ablations are *controls*; do not count them as extra checkpoint arms.
5. Keep raw model responses, vectors and weight files local. Never enable serving or publish the edited weights without separate permission and safety checks.

## Frozen provenance

- A SHA-256: 135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330
- A Hugging Face revision on local source: b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf (source revision file; checksum governs exact bytes)
- C upstream revision: 545fc8f4c7b5fa9f405105274cd29c4f8f44b526
- B source, vector and output SHA-256: in private RESEARCH_PROVENANCE.json and sanitized evaluation evidence.
- Warning: currently only TWO calibration pairs; this is a hardware/integration smoke test, not a scientifically valid effect estimate.

## Evaluation requirements before publication

- Freeze paired train/validation/heldout splits and report all sample counts.
- Add a substantially larger domain-balanced corpus, multiple seeds, blinded independent human review, and inter-rater agreement.
- Evaluate both benign utility and adversarial non-compliance, not just refusal keywords.
- Test sandboxed symbolic tool requests under strict authorization; no production effects.
- Check post-normalization reintroduction of refusal direction in the LFM3 hybrid stack.
- Disclose uncertainty intervals, negative results, prompt leakage checks and failure cases.
- Keep PR as draft until independent review. No model-weight release, arXiv submission, or merge implied.

## Smoke commands

On x1-370, under the research study directory:

- python -m pytest -q tests
- python -m src.build_abliterated_checkpoint --allow-smoke --layer 10 --alpha 1
- python -m src.three_checkpoint_eval --max-heldout-pairs 2 --max-new-tokens 20

The second command intentionally refuses to overwrite an existing checkpoint.

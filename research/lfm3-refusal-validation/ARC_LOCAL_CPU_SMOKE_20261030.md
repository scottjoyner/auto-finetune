# Three-checkpoint LFM3 CPU ARC smoke, October 30, 2026

**Real lm-eval 0.4.13 scoring completed; ONLY first 40/1172 test questions per checkpoint.** Not the official full benchmark, representative sampling, proof of retained capability, zero-refusal, or safety.

| Frozen model arm | acc | acc_norm | Number correct (acc / acc_norm) |
|---|---:|---:|---:|
| A original post-trained | 20.0% | 25.0% | 8/40 / 10/40 |
| B M1 rank-one derivative | 17.5% | 25.0% | 7/40 / 10/40 |
| C pretrained base | 40.0% | 40.0% | 16/40 / 16/40 |

Actual three-arm execution: 160 official-format multiple-choice loglikelihood requests per arm, 480 total. Zero-shot, CPU bfloat16, batch size 1, identical fixed seeds and ordered questions. B minus A: -2.5 percentage points unnormalized accuracy, 0.0 percentage points normalized accuracy. Insufficient for paired significance or equivalence.

The canonical built-in ARC task initially failed because the EleutherAI ARC dataset was missing from the isolated offline Hugging Face cache. We used the same canonical question/answer prompt, choice ranking, acc and acc_norm rules inherited from official arc_easy.yaml, substituting only a pinned local test parquet using a PRIVATE task adapter named arc_challenge_local_pinned. The task overrides dataset transport, dataset_name and unavailable train/validation splits. Accordingly the scores are LOCAL ADAPTED SUBSET measures, not canonical official leaderboard results.

x1-370 Radeon 890M model-loading worked but the ROCm embedding kernel failed with HIP invalid device function. CPU fallback completed for all original local safetensors checkpoint arms. No weight edits, model copies, GPU driver changes, public generations, release or deployment occurred.

The PUBLIC aggregate report in evidence/20261030_arc_local_first40_cpu_aggregate.json contains:
- frozen checkpoint and test parquet SHA-256 identities
- private task YAML and raw result JSON SHA-256 fingerprints without private content
- complete scoring configuration, scores and their standard errors
- explicit limitations and release-denial fields

Original raw results and private local task YAML stay on x1-370 under results/lm_eval_arc_local_private and are excluded from GitHub. To repeat this limited validation, use lm-eval v0.4.13 with --include_path pointing to the local private adapter, --tasks arc_challenge_local_pinned, --num_fewshot 0, --limit 40, --batch_size 1, --device cpu and the exact frozen model path. The first-40 limit must be removed for the future separately authorized full 1172-question benchmark, whose strict official-result gate remains closed.

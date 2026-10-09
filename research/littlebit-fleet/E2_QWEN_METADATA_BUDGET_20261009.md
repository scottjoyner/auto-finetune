# E2 preflight — Qwen3-0.6B metadata-only physical BPW budget

**2026-10-09 | Stage: E2 PREFLIGHT ONLY | Pretrained model weights: NOT_DOWNLOADED | Training: NOT_STARTED**

This work is stacked on [E1c official CPU class PR #19](https://github.com/scottjoyner/auto-finetune/pull/19) and tracked by issue #17.

## Exact model metadata and rights

- Model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).
- **Pinned revision:** [`167b8104f88905a951069f5f95f9776908da5f68`](https://huggingface.co/Qwen/Qwen3-0.6B/blob/167b8104f88905a951069f5f95f9776908da5f68/config.json).
- Fetched **only** the 726-byte `config.json` on x1-370; SHA256 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
- Model card currently identifies Apache-2.0, but any combined use with LittleBit's CC BY-NC 4.0 upstream remains subject to explicit license review.
- Configuration: `Qwen3ForCausalLM`, 28 layers, hidden 1024, 16 Q heads at head dimension 128, 8 KV heads, MLP width 3072, vocabulary 151936, `tie_word_embeddings=true`.
- No safetensors, checkpoints, tokenizers or pretrained weights were downloaded or loaded.

## Projection definition

This experiment is a **closed-form storage budget**, NOT measured checkpoint or compression quality.

For each of the seven assumed transformer linears per layer (Q/K/V/O and gate/up/down), consider ranks **multiples of 8**. Choose the largest rank whose **projected upstream-style** tensor payload stays under each requested BPW. Count int32 32-bit-row-aligned packed factors, four FP32 scale vectors per branch, two int64 shape tensors and three registered scalar buffers. Assume one primary branch, **no residual**, no linear biases; this is the E1c float32 scale/storage path and may differ for BF16 cast checkpoints.

The 28-layer transformer linears total **440,401,920** original parameters. Qwen3's tied embedding matrix contains **155,582,464** parameters, assumed retained at **BF16 / 2 bytes per element**. Projected denominator excluding norms and other small tensors is **595,984,384 parameters**. Do not divide compressed block bytes by whole-model params or silently count the tied embedding twice.

## Observed metadata-only computed projections

| Target transformer-linear BPW | Attained estimated linear BPW | Whole model BPW (BF16 tied embeddings, excludes norms/ZIP) | Tensor bytes estimated |
| --- | ---: | ---: | ---: |
| 0.55 | 0.5416097005 | **4.5770416964** | **340,980,672** |
| 0.30 | 0.2932047526 | **4.3934831688** | **327,305,920** |

For the 0.55 candidate, chosen per-module ranks are `q=320, k=224, v=224, o=328, gate=384, up=384, down=384` (out/in padding can make transpose-shaped layers cost differently). For the 0.30 candidate: `q=160, k=104, v=104, o=160, gate=192, up=192, down=192`.

**Not established:** whether upstream's automatic rank selection chooses these ranks; whether every listed module is converted by its actual Qwen3 adapter; bias/norm tensor count; serialization overhead; real whole-checkpoint BPW; model quality; any speedup. Keep FP32 scales and BF16 tied embedding as explicit assumptions.

## Exact reproducibility / validation

From the research branch, using **stock Python 3.12.3** and the pinned 726-byte local config (no torch):

```bash
LITTLEBIT_QWEN_CONFIG=/home/scott/git/littlebit-qwen3-config-20261009.json \
  python3 -m unittest discover -s tests \
    -p test_littlebit_qwen_metadata_budget.py -v

python3 -m experiments.littlebit.qwen_metadata_budget \
  --config /home/scott/git/littlebit-qwen3-config-20261009.json \
  --inspect-pinned-metadata > /home/scott/git/littlebit-qwen-budget-20261009.json
```

**x1-370: 9/9 tests PASS**, including exact config SHA, BPW math, infeasible small matrix, two-branch accounting, malformed metadata, disarmed command and explicit tied embedding. Complete JSON output SHA256: `149aeca37c72c5cba1410bdb19cb8b45ecc3c316bb3c6e5f1708753b0fd5963c`. Original output is retained locally at the x1 path and **not claimed to be a GitHub artifact**.

## Gate assessment and next decision

**E2 model-metadata planning: PASS.** **E2 actual model conversion: HOLD.** This data makes a compelling systems question: if BF16 embeddings dominate whole-model size, does sub-1-bit linear compression justify its training cost relative to regular Q4, or should we separately test embedding quantization?

The narrowest next acceptance is a **read-only module-path compatibility proof on a pinned upstream Qwen3 architecture**, including quantizer target-exclusion behavior and per-layer dtype constraints; only after successful review should a licensed, capacity-bounded model-weight download and single-layer CPU compression be authorized.

No GPU or hosted provider work, no commercial/production use, and no NAS write is authorized by these results.

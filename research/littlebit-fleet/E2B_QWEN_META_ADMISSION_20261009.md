# E2b — real upstream Qwen3-0.6B META module mapping and byte-accurate admission

**2026-10-09 | Research only | STAGE E2b | Status: module topology verified, nominal rank admission DENIED.** Tracks #17, stacked on Qwen storage estimate PR #20 and real LittleBit CPU class PR #19.

## Method and exact custody

- **No model weights or tokenizer downloaded.** Only the pinned **726-byte public Qwen3-0.6B config** from `Qwen/Qwen3-0.6B@167b8104f88905a951069f5f95f9776908da5f68`, SHA256 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
- Qwen3 architecture instantiated with `torch.device("meta")`: **no real weight tensor storage**. Torch 2.6.0+cpu, transformers 4.51.3, Python 3.12.3, verified in isolated `/home/scott/git/littlebit-parity-venv-20261009` on x1-370. Its version is pinned by the local runtime check; not yet an independently reviewed lockfile for other hosts.
- Real `SamsungLabs/LittleBit@42d658b0c79f76450b34b6a3547462c7cdc3e1a0` `quantization.utils.quant_util.apply_littlebit_patch` imported from separate read-only checkout. Checked source file Git blob IDs: `quant_util.py=c40a35168ab8f02411dfcd484f0fb4ae11b026b9`, `littlebit.py=43ce9d2f9383b676c3898346ece2d3994269a908`, `binary_packer.py=78f0e20f8a525d3cdb0f94908fd767077834cf4b`. The upstream source is **not vendored** in this PR.
- Forced `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=''`. CPU Torch and config SHA verified before running; no training CLI, provider calls, checkpoint mutation, WAN API, NAS writes or production scheduler.
- `apply_littlebit_patch(do_train=False)`: actual upstream class conversion of meta layers, **not** weight compression or initialization, and no meaningful quality or decoding performance result.

## Real architecture / conversion acceptance

The unmodified `Qwen3ForCausalLM` from pinned config has **197 `nn.Linear` modules**: seven projections (Q/K/V/O and MLP gate/up/down) across 28 decoder layers, plus `lm_head`. All seven have the exact output/input dimensions predicted by the metadata planner; `attention_bias=false` and no projection bias is present.

Actual pinned patch `mapping={nn.Linear: LittleBitLinear}` converts **196/197**. Exactly `lm_head` remains excluded. Input/output embeddings remain **one tied parameter**, so whole-model storage accounting must not double count them. 196 quantized modules and all retained parameters are still on `meta`; no actual weights exist.

## Deny-only rank admission

Our new read-only policy computes projected physical tensor bytes based on upstream's **actual selected rank**, with 32-bit row-padded binary sign tensors, four FP32 branch scale tensors, two int64 shape tensors per branch, three scalar buffers per layer, plus optional second residual branch. A target means *actual tensor BPW <= target*, **not** upstream `_compute_eff_bits`. This policy **does not mutate any rank or model**; the future rank override/adapter requires a separate engineering gate.

| Upstream target | Branches | Example selected ranks (q / k / v / o / gate / up / down) | Real patch conversions | Byte-limit violations | Projected linear BPW | Projected linear bytes |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.55 | 1 | 352 / 256 / 256 / 352 / 400 / 400 / 400 | 196 | **196** | **0.5797932943** | 31,917,760 |
| 0.30 | 1 | 184 / 136 / 136 / 184 / 208 / 208 / 208 | 196 | **196** | **0.3329182943** | 18,327,232 |
| 0.55 | 2 | 168 / 120 / 120 / 168 / 192 / 192 / 192 | 196 | **196** | **0.6065348307** | 33,389,888 |

The independent byte-feasible 0.55 BPW `q_proj` rank is **320**, versus upstream-selected **352**. For 0.30 BPW: feasible **160**, selected **184**. These are examples of the intervention future engineering must investigate; **no override was applied in this experiment**.

The projected selected-tensor storage including one **uncompressed tied BF16 embedding** (311,164,928 bytes), and still excluding norms/ZIP/tokenizer/auxiliary tensors:
- Upstream nominal primary 0.55 => ~343,082,688 bytes, **4.6053** partial whole-model BPW.
- Upstream nominal primary 0.30 => ~329,492,160 bytes, **4.4228** partial whole-model BPW.
- Upstream nominal residual 0.55 => ~344,554,816 bytes, **4.6250** partial whole-model BPW.

These are **analytical byte projections of inspected meta shapes**, not measured `torch.save` checkpoint sizes; FP32 scales follow the actual E1c initialization-path observations but could be BF16 after explicit cast in final export. The real model total also contains norm tensors and additional metadata. Do not describe projected byte savings as realized on disk.

## Evidence and tests

**x1-370: 12/12 focus tests pass**, comprising nine standard-library policy/source/negative tests and three actual upstream meta-integration configurations. One attempt initially failed because the upstream `quantization` package remained cached across sequential audits; isolated import-cache cleanup was added and tests rerun successfully.

Hash-addressed JSON of all **196 per-module decisions per configuration** on x1 local SSD:
- `/home/scott/git/littlebit-meta-admission-055-20261009.json`, SHA256 `f7309eee9f3790f4671390331301deb6f0fc18324d9010b0aaeb617ceb523c0e`.
- `/home/scott/git/littlebit-meta-admission-030-20261009.json`, SHA256 `c895005dcb904c08c79c373bf699faf1b015cf5a060df34beac165f85c6b233b`.
- `/home/scott/git/littlebit-meta-admission-055-residual-20261009.json`, SHA256 `b24ba7207eb030ee4f9ab952313a485fc59a57599cfd867dfcfc7cc9a55fd215`.

Tests:
```bash
cd /home/scott/git/wt-littlebit-qwen-meta-admission-20261009
LITTLEBIT_QWEN_CONFIG=/home/scott/git/littlebit-qwen3-config-20261009.json \
LITTLEBIT_UPSTREAM_ROOT=/home/scott/git/littlebit-upstream-audit-20261009 \
LITTLEBIT_RUN_META_TEST=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
/home/scott/git/littlebit-parity-venv-20261009/bin/python \
  -m unittest discover -s tests -p test_littlebit_qwen_meta_admission.py -v
```

GitHub's separate `LittleBit research metadata CPU` workflow tests stdlib rank rules without Torch. The real meta tests require the separate pinned Torch/upstream environment and **are not silently counted as passing hosted CI**. The whole repo pytest workflow has unrelated missing-dependency collection blockers.

## Decision

**E2b module selection: PASS. E2b physical BPW admission: DENY on all nominal candidates tested.** Both outcomes are required to be reported together. The intended 0.55 target is **not met** by upstream's automatic ranks under our validated FP32-scale representation.

**Next smallest reversible engineering slice:** a non-mutating per-module `rank_plan` override **proposal** with a strict physical-bit budget, verification that the *actual upstream conversion entry point* can accept explicit per-module ranks, tests for rank-fallback/infeasible dimensions, and independent packed tensor byte verification. DO NOT insert a planner silently into QAT/model serving. Separately confirm Qwen/model and LittleBit source licensing and whether export casts scales to BF16; only then consider a capped, reviewed pretrained-weight single-layer experiment.

**All current gates for pretrained weight downloads, QAT, hosted inference, GPU training, NAS writes and production model promotion remain HOLD/DENY.**

## E2b.1 refinement — pinned active BF16 export path (2026-10-09)

**Important correction to interpreting the above FP32 table:** the pinned upstream `main.py` (Git blob `d5f68106ed78e103eaa05c6feda77ce66c3bad03`) defines `save_artifacts` **twice**. The **second (effective)** definition applies `float32 -> bfloat16` to state-dictionary tensors unless their keys contain `packed` or `shape`, before calling `save_pretrained(..., safe_serialization=True)`. Thus the FP32 initializer-state projection above is **not equivalent to the active main.py export projection**. The export also converts the two FP32 quantization metadata buffers, saving four additional bytes per module beyond casting the four branch scale tensors. Other upstream export entry points may differ.

The deny-only gate now reports the two storage cases separately, using actual ranks returned by the official `apply_littlebit_patch(do_train=False)` meta execution:

| Upstream nominal setting | Initializer-state FP32 BPW | Initializer-state modules over target | Active main.py BF16-export *projected* BPW | Export-projected modules over target | Export-projected linear tensor bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.55, one branch | 0.57979329 | 196/196 | **0.55194702** | **84/196** | **30,384,816** |
| 0.30, one branch | 0.33291829 | 196/196 | **0.30741577** | **140/196** | **16,923,312** |
| 0.55, two branches | 0.60653483 | 196/196 | **0.55599976** | **112/196** | **30,607,920** |

At primary 0.55, all 84 export-budget violations are in `mlp.gate_proj`, `mlp.up_proj`, and `mlp.down_proj` (28 layers each); the four attention projections pass under the BF16-export assumptions. At primary 0.30, 140 violations cover the two MLP expansion projections and the Q/K/V attention projections, while O and MLP down meet the target. At residual 0.55, 112 violations cover the four attention projections and the MLPs meet the target.

**Independent synthetic cast check:** using the actual pinned `LittleBitLinear` class on an isolated CPU 128×256/rank16 matrix, we inspected its `state_dict()`, applied the active export's explicit cast predicate without importing or invoking the upstream training CLI, and confirmed its true PyTorch tensor byte totals match our FP32 and BF16-export arithmetic exactly, including preserved int64 shape and packed int32 tensors. This is not a production checkpoint or confirmation of the final `save_pretrained` container size.

**Latest x1 tests:** 14/14 focus tests pass (ten stdlib rank/source/negative tests and four separately opted-in actual Torch/source integration checks), including sequential audit import isolation and the synthetic `state_dict` export-cast parity check. The earlier 12/12 results describe a prior branch revision.

**New immutable local evidence files**, containing 196 per-module FP32 and BF16 export decisions per setting:
- `/home/scott/git/littlebit-meta-dual-budget-055-20261009.json`: SHA256 `e28da4ea16d022a1d536e7108cecf7e1ee76f0a2f524a49547f211b0a48b1e26`.
- `/home/scott/git/littlebit-meta-dual-budget-030-20261009.json`: SHA256 `395fbd9a802cb7811b877b5c00266590a5b4d8498e08449020e7207abea14e86`.
- `/home/scott/git/littlebit-meta-dual-budget-055-residual-20261009.json`: SHA256 `71251ff04002a799b4025736d2d37b69c8215fd7df2dfc727c3c204d19693b2a`.

**Gate remains DENY** for all three tested nominal targets even under BF16 export assumptions, because aggregated and some per-module byte budgets exceed the target. Nothing changes ranks, pretrained weights, QAT, deployment, or fleet services. Future work should test a separately reviewed per-module rank plan and actual *full* checkpoint serialization, without conflating state-dictionary payload with a safe deployed format.

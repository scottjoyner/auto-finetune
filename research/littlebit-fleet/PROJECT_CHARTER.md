# Project charter and architecture decision record

**Draft:** 2026-10-09 | **Status:** proposal awaiting review | **Tracking:** [#17](https://github.com/scottjoyner/auto-finetune/issues/17)

## Research problem and decision

Sub-1-bit storage does not automatically produce small *training* cost or fast deployed inference. QAT can be dominated by teacher activation retention, student master weights, optimizer state and non-binary kernels. We therefore require **three ledgers**: (a) compressed storage, (b) training runtime/VRAM/energy, (c) deployed inference quality/latency.

The central conjecture: once a weight matrix is decomposed into binary low-rank factors and learned row/column/latent scales, *a meaningful fraction of QAT's post-compression quality recovery may live in the much smaller scale subspace*. Optimize that subspace with exact autograd first; compare against antithetic forward-only optimization **only after** the exact-gradient case improves genuine held-out quality.

## Goals and ranked deliverables

1. Independent CPU-oracle for factorized linear and primary+residual representation, including exact packed storage accounting.
2. Reproducible Dual-SVID and Joint-ITQ initialization A/B test on fixed synthetic spectra and an upstream-supported public reference model.
3. Teacher/student QAT contrast: no-QAT / scales-only / full upstream QAT with budget- and model-matched arms.
4. Calibration of true local gradients versus orthogonal/gaussian antithetic estimators for the **same** final-layer scale vector.
5. On-target device comparison of quality, checkpoint size, RAM/VRAM, tokens/sec, p50/p95 TTFT and decoding latency.
6. Evidence-first publication with all preregistered negative experiments and environment information.

## Explicit non-goals

- Not a zero-loss 0.1 BPW claim; 0.1 BPW is a small, isolated cliff demonstration and is never a release target.
- Not a full Dust reimplementation or a proof that gradients everywhere can be replaced.
- Not source conversion of K2-Horizon, Qwen3.5, Bonsai/ternary formats, or existing local models **until architecture-specific parity is proved**.
- Not use of private user-agent logs or production prompts in public reports.
- Not running a fleet scheduler, writing to authoritative NAS5, modifying production API/assistant services, deploying weights, or enabling commercial use under a noncommercial license.
- Not a promise of a performance gain; if a smaller model gets slower, record that as the result.

## Fleet resource plan (tentative, not allocation)

| Target | Intended role | Condition |
| --- | --- | --- |
| Desktop / X1 CPU | E0/E1 independent arithmetic, documentation and no-GPU tests | CPU and available memory confirmed; no service interruption |
| Xwing ROCm | Optional diagnostic portability comparison | Exact runtime compatibility established; dedicated idle window |
| RTX3090 host | Preferred CUDA pilot candidate | Must locate working physical GPU, CUDA version, usable VRAM, service owner approval |
| R9700 | Separate emerging runtime candidate | Accept only after physical gfx1201 and exact pinned software proof |
| MacBook Air | Documentation/review and CPU smoke when convenient | No assumed CUDA support |
| Beelink/NAS5 | **No project training/data target** while capacity/recovery gates remain open | Never write checkpoints/experiments to recovering volume |

No training hardware has been validated or reserved by this project yet. All model downloads, memory footprint and artifacts require preflight free-capacity checks and storage ownership verification. Default to local SSD working directory; never silently spill to a NAS mount or fallback directory.

## External compatibility and license requirements

Official upstream: https://github.com/SamsungLabs/LittleBit ; paper: https://arxiv.org/abs/2506.13771 ; newer initializer: https://arxiv.org/abs/2603.00042.

- Pin **commit hash**, license snapshot, paper version, model ID and revision, framework lock and tokenizer artifacts.
- Upstream READMEs currently recommend torch 2.8.0+cu124 but requirements.txt pins torch 2.6.0: audit and select a tested environment, do **not** unconditionally install both.
- Upstream is stated as **CC BY-NC 4.0**. Establish boundaries for reuse, redistribution, derivative code and any production use with appropriate review. Experimental independently authored CPU oracle may coexist, but must not misrepresent upstream origin.
- Do not treat a tested small-model forward as CUDA-kernel, speed, or model-scale compatibility.

## Risk register (owner assignments pending)

| Risk | Impact | Preflight / mitigation | Hold condition |
| --- | --- | --- | --- |
| Quantization cliff | Poor reasoning/tool behavior | 0.55 first; compare heldout and failure modes | No positive quality recovery |
| Incorrect BPW reporting | Invalid scientific claim | Bit-pack estimator plus serialized bytes and whole-model denominators | Accounting or decode mismatch |
| Teacher QAT memory spike | Fleet OOM/service interruption | Dry-run estimate, batch=1 smoke, hard VRAM and RAM floors | Memory margin cannot be met |
| Hidden data leakage | Inflated heldout results | Hash/dedupe at document/task and lineage level before tokenization | Train/heldout contamination |
| CUDA vs ROCm divergence | Incorrect speed claims | Same hardware/kernel for pairwise results, capability reports | Kernel unsupported |
| Unintended deployment | Production behavior changes | Separate opt-in experiment namespace; deny-by-default run config | Any publish/serve/scheduler permission |
| License conflict | Redistribution/commercial issue | Record upstream license and model/dataset permissions | Review not cleared |
| GPU scheduling contention | Lost concurrent work | Snapshot running services and reserve idle window | Active fine-tune, NAS pressure or inference demand |

## Governance and audit

Every run must have an immutable run ID, parent experiment version, prediction signed/time-stamped **before** execution, source/revision digests, host owner/consent, job lease/budget, source data digest, model/tokenizer digest, observable stopping decision, output hashes and measured vs predicted row. Keep high-sensitivity raw prompts, secrets and telemetry out of git and research manifests. Require review before expansion across GPU, datasets, host nodes, and checkpoints.

**Initial gate:** E0/E1 only. A research PR may merge documentation and standalone CPU tests without granting any training/deployment authority.

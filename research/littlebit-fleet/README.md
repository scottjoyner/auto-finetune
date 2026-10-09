# LittleBit-Fleet — research project (DRAFT)

**Date:** 2026-10-09 | **Primary tracker:** [auto-finetune #17](https://github.com/scottjoyner/auto-finetune/issues/17) | **Status:** DESIGN ONLY / ALL EXECUTION DISARMED

## Mission

Test whether LittleBit's sub-1-bit factorized representation, plus **scales-only adaptation** and optionally **forward-only antithetic updates**, can retain or recover useful model capabilities with substantially lower *training resource cost* than full quantization-aware training (QAT). **Compression, training efficiency, and inference throughput are separate outcomes**; one does not imply the others.

The canonical experiment compares upstream LittleBit Dual-SVID against LittleBit-2 Joint-ITQ, both at equivalent *measured* effective bits per weight (BPW), identical model/data/optimizer budgets, with a matched no-QAT control. A later, separate experiment asks whether the cached-tail, orthogonal estimator used in this repository's Dust work can optimize LittleBit scale parameters.

## Read in order

1. [PROJECT_CHARTER.md](PROJECT_CHARTER.md): aims, boundaries, infrastructure, license and risk register.
2. [HYPOTHESES.md](HYPOTHESES.md): preregistered null hypotheses, endpoints, success and stop criteria.
3. [DESIGN.md](DESIGN.md): layer math, interfaces, model coverage, storage accounting and provenance.
4. [EXPERIMENTS.md](EXPERIMENTS.md): predeclared experiment matrix, controls, reproducibility and statistical plan.
5. [DELIVERY.md](DELIVERY.md): work packages, dependencies, gate owners and handoff requirements.
6. [config/research-defaults.yaml](config/research-defaults.yaml): **deny-by-default configuration template**, not a live job definition.
7. [papers/littlebit-fleet/main.tex](../../papers/littlebit-fleet/main.tex): independent scientific-report draft. Never put unmeasured numbers in the Observed column.

## Scope separation

- Main working repository is **auto-finetune**. All new tooling must be opt-in under research/littlebit-fleet or later isolated scripts/tests; do not call existing scheduler, deploy, merge, quantize or registry paths.
- [Dust/K2 PR #13](https://github.com/scottjoyner/auto-finetune/pull/13) is an independent reference, **not** a dependency branch, approved production estimator or a positive held-out generalization result.
- [Tunix/Jev PR #16](https://github.com/scottjoyner/auto-finetune/pull/16) is a separate offline feasibility investigation and is not a live RL back end for this project.
- **No private Opencode/Hermes session data** in phase E0/E1; no inference endpoint calls, checkpoint promotion, model serving, bulk NAS transfer, GPU preemption or commercial integration.

## Authoritative external sources

- [Lee et al., *LittleBit: Ultra Low-Bit Quantization via Latent Factorization*, NeurIPS 2025 / arXiv 2506.13771](https://arxiv.org/abs/2506.13771).
- [Lee & Kim, *LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment*, ICML 2026 / arXiv 2603.00042](https://arxiv.org/abs/2603.00042).
- [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit), **CC BY-NC 4.0** as labeled in its README. Pin source commit and resolve permitted use before importing upstream code.
- The original paper reports its 0.3–0.55 BPW quality/size tradeoff on particular model/data/hardware configurations. These are motivations, **not predictions for K2, Qwen3.5, Bonsai or fleet throughput**.

## Quick orientation: gate sequence

E0 *sources / licensing / hardware read-only* → E1 *CPU matrix / arithmetic* → E2 *upstream-compatible 0.55 BPW GPU smoke, separately approved* → E3 *multi-seed quality / baselines* → E4 *forward-only scales experiment, separately approved* → E5 *on-device inference and report*.

Each gate has a hard FAIL/HOLD option. Document a negative result; do not automatically increase compute to recover a preferred conclusion.

### Current facts (not project results)

- The upstream README currently recommends Python 3.12, CUDA 12.4, PyTorch 2.8.0+cu124, transformers 4.51.x; its requirements.txt separately pins torch==2.6.0 and transformers>=4.51.0. **Resolve this conflict with a reproducible lock** before running E2.
- Upstream currently lists support for Qwen3 (also Qwen2.5, Llama, OPT, Phi-4, Gemma). Support for K2-Horizon/Bonsai is **not established**.
- Paper's A100 CUDA kernel results do not establish ROCm, Vulkan, RTX3090 or Strix Halo performance.
- Full-model checkpoint compression is not the same as block-level BPW: embeddings, norms, lm_head, scales, tensor packing, metadata and residual branches all count in true storage.

**Until further approval:** `research_only: true`, `gpu_training_enabled: false`, `allow_deployment: false`, `allow_external_provider_calls: false`.

## Source snapshot pinned for reproducible E0 review

- Upstream Git commit (snapshot inspected on 2026-10-09): [`42d658b0c79f76450b34b6a3547462c7cdc3e1a0`](https://github.com/SamsungLabs/LittleBit/commit/42d658b0c79f76450b34b6a3547462c7cdc3e1a0). Treat the reference as an **unreviewed snapshot**, not yet a validated build.
- The pinned README is confirmed at that ref; README blob SHA `0a480659fcdf987e8a32a81aacf864580ee380ac`. Upstream `requirements.txt` blob inspected on main: `f99190c4edc02bc3e4bc262249fb3ffb9f93628b`; confirm that blob matches the pinned commit when resolving lock.
- No dataset or model checkpoint revision, licensing approval, host authorization, training result or dependency lock is yet recorded.

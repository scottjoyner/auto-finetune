# Experimental protocol and acceptance matrix

**Version:** 0.1 DRAFT, 2026-10-09. Metrics here are design targets, not observations. **Freeze protocol before running E1.**

## General rules

- Unit of analysis is a model/data/hardware/seed/bit-budget matched run; pair treatment and control whenever feasible.
- Record seeds 7, 42, 1337 and software/hardware digests, never move samples between train and evaluation.
- Separate `calibration` (find estimator behavior), `validation` (choose checkpoints/hyperparameters), `heldout` (gate comparison), and `final_test` (one untouched reporting pass).
- Predeclare selected target matrix shapes, spectra and model modules before seeing quality results. For E1 use, provisionally, 3 shapes (64x64, 128x256, 256x128) x 3 spectra (low rank, flat, heavy tail); no file I/O beyond artifacts to approved temporary local SSD.
- Strict equality: *same* rank/bit budget **including primary+residual scales**. Report full cost of extra initialization and the backend's true training memory.
- No uncontrolled adaptive search: exploratory changes are new labeled experiments, and final_test is never used for tuning.

## Stage E0 — source/reproduction preflight (no model load)

**Planned work:** Check paper and official repository; record exact upstream commit, source license, Qwen3 supported path and tokenizer config, dependency conflict, CUDA/ROCm availability, device occupancy and storage. Confirm command allowlist and scope.

**Evidence:** signed source/license table; dependency lock; compatibility matrix; predicted GPU resource envelope; fail-closed execution config. **GO** only if license use and local resource path are supportable; else HOLD. No installs/downloads without explicit scope.

## Stage E1 — synthetic independent CPU oracle

| Arm | Init | Residual | Rank/BPW | Target metric |
| --- | --- | --- | --- | --- |
| E1-A | FP32/FP16 full matrix | n/a | baseline | near-zero oracle error |
| E1-B | naive SVD then binarize | off | matched | reconstruction error |
| E1-C | Dual-SVID | off | matched | reconstruction error |
| E1-D | Dual-SVID | on | combined matched | reconstruction error |
| E1-E | Joint-ITQ | off | matched | reconstruction error |
| E1-F | Joint-ITQ | on | combined matched | reconstruction error |

Targets block BPW 1.0, 0.55 and 0.30 where dimensions permit. **Do not force unattainable settings:** for small shapes FP16 scale overhead may exceed the bit budget; record `INFEASIBLE`, use larger predeclared synthetic matrices if appropriate, or test rank-matched pairs. This impossibility is itself important evidence.

Test dense-vs-factor forward equality, rank arithmetic, primary/residual budget allocation, numerical stability, gradients of scales and frozen-mask assertions, serialized sign bit packing/unpacking, byte order, deterministic seed, graceful OOM/error paths and negative network/deploy actions. Outcome: paired mean and per-seed reconstruction error, output cosine, CPU memory/runtime and packed bytes. **No claim about model perplexity** from E1.

## Stage E2 — one CUDA/compatible public-model pilot, separate approval

**Precondition:** E0+E1 witnessed GO; actual upstream target Qwen3-0.6B revision verified; selected node idle and compatible; time+VRAM+disk caps approved.

**Controls at 0.55 BPW:**
- P0 frozen full-precision teacher/float baseline.
- P1 Dual-SVID no QAT.
- P2 Joint-ITQ no QAT.
- P3 Dual-SVID full factor+scale QAT.
- P4 Joint-ITQ full factor+scale QAT.
- P5 Joint-ITQ scales-only QAT.
- P6 optional Joint-ITQ latent-only QAT if preliminary scales mask validated.

Up to 100 steps, matched batch/sequence/tokens, fixed teacher objective (output KL + 10 x inter-layer MSE) and model weights; isolate CPU and GPU memory incl teacher. Predeclare the actual learning-rate and any differential optimizer policy in a versioned manifest **before** measurements. An LR that is stable for full QAT may not be stable for scales-only; use a small, explicitly separate training-only calibration set if selecting LR, never use heldout/test to tune.

Stop at first OOM, nonfinite parameters, split/hash change, gateway call, unexpected model write or resource budget breach. **GO** to E3 only if at least one preregistered treatment shows lower heldout CE than its untrained counterpart, and audits pass. Mark pilot as *exploratory* even if positive.

## Stage E3 — multi-seed quality and bit-budget sweep

Only after successful pilot; run seeds 7/42/1337 on best *predeclared* arms with independent evaluator and frozen heldout. Report per-run CE and task success, median/range and paired differences. Run public general reasoning and coding/tool correctness probes with fixed prompts, deterministic generations and exact rubric. Add 0.30 BPW after a separate experimental gate; 0.10 is a bounded synthetic cliff test only unless a new preregistration justifies more.

Follow a control ladder: BF16/FP16, real Q4 (matching model + generation), no-QAT LittleBit, scales-only and full-QAT. No claim that LittleBit beats Q4 merely because nominal bits are smaller. Ensure `test` examples are neither distilled by the teacher nor in calibration.

## Stage E4 — isolated forward-only study (new hypothesis)

**Precondition:** E3 establishes scales-only autograd improvement. A sign-factor-frozen, final-layer or bounded cached-tail scale-vector probe must have an independently checked exact local gradient oracle.

| Treatment | Forward count | Backward | Primary comparison |
| --- | --- | --- | --- |
| Exact local scales autograd | report | yes (control only) | gold gradient |
| Orthogonal antithetic scale update | matched | zero | cosine/error, heldout CE |
| Gaussian antithetic scale update | matched | zero | variance control |
| Shuffled or wrong-sign control | matched | zero | invalid-gradient detection |
| No update | minimal | zero | natural metric drift |

Separate prefix-cache construction time from tail replay and from all-GPU model-forward equivalents. Report estimator quality vs *dimension of scale vector*, K, smoothing sigma, directions per batch and number of scored positions. The K=1024/sigma=0.25/D=4 K2 reference is calibration history, not a fixed optimum for LittleBit. Choose any new parameters on training-side oracle and then **freeze** them for heldout evaluation.

Do not enter 8/16/32-update scaling without a new witnessed gate showing actual heldout improvement. Never rename a 2-step synthetic smoke an end-to-end training success.

## Stage E5 — deployment-independent benchmark and paper

Inference device capability matrix includes kernel support (not just model-load success), packed tensor footprint, model cache, TTFT, separate prefill/decode TPS, batch=1, several prompt lengths and thermal drift. Stable decode checks: logits parity against a dense oracle where tractable and deterministic output repetition. Repeat 10+ trials paired on *same physical node*. Place benchmark summaries in immutable public-safe evidence; keep model files on an approved location and **do not deploy to shared service**.

## Budget and negative acceptance

| Gate | CPU/GPU ceiling | Persisted artifacts | Failure behavior |
| --- | --- | --- | --- |
| E0 | Read-only local checks | source/env evidence | HOLD without pin/license |
| E1 | CPU only, synthetic | fixtures + manifest + tests | FAIL for byte or oracle mismatch |
| E2 | GPU ceiling *to be approved*; <=100 steps first | small research-only evidence | kill job and preserve partial audit |
| E3 | explicit per-seed authorization | paired score summaries | publish negative/no-go |
| E4 | strict model-forward budget, isolated | exact-gradient comparison | no escalation on gradient cosine alone |
| E5 | no production services | measured device reports | no performance extrapolation |

**Default budgets are 0 GPU seconds and 0 provider tokens until approved.** Do not quietly spend a provider free tier. Budget envelopes (GB VRAM, GiB disk, watt-hour, wall time) must be filled from a physical preflight; unspecified budget means DENY.

## Results ledger (do not prefill)

| Run ID | Prediction committed | Actual hardware | Actual block BPW | Actual file bytes | Heldout CE | Task success | Memory | Wall time | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOT_RUN | planned | unverified | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | HOLD |

The manuscript's first submission must include all failed or inferior arms and a section titled Threats to Validity (e.g., limited seeds, model scale, noisy energy, kernel support, test leakage and upstream drift).

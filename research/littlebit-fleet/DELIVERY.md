# Delivery plan, milestones and dependencies

**Status 2026-10-09:** design tasks drafted, NOT assigned to executors. All hosted/free provider and fleet execution remain disarmed unless separately approved and traced.

## Critical-path milestone ladder

| Milestone | Work package | Concrete deliverable | Required witness | State |
| --- | --- | --- | --- | --- |
| M0 | Research grounding | Citation/license/dependency/model inventory + pinned revisions | Reviewer of source and rights | DRAFT |
| M1 | Math/serialization oracle | Dense-vs-factor parity, scale derivative tests, packing and exact BPW tests | Independent CPU test output | NOT_STARTED |
| M2 | Initializer fidelity | Dual-SVID vs ITQ / residual / seed matrix and prediction-observation report | Repro manifest + hashes | NOT_STARTED |
| M3 | Model feasibility | Qwen3-0.6B verified model layers and compatible resource envelope | Node owner / runtime witness | BLOCKED_ON_E0_E1 |
| M4 | QAT 100-step smoke | paired no-QAT/full/scales arms, clean heldout, measured memory | Independent eval + run trace | BLOCKED_ON_GPU_AUTH |
| M5 | Quality validation | multiseed 0.55 BPW paired study; 0.30 conditional | Independent evaluation report | BLOCKED_ON_M4 |
| M6 | Orthogonal update study | exact-vs-estimated gradient & after-update heldout CE | Repeated witness, per-forward accounting | BLOCKED_ON_M5 |
| M7 | Deployment-independent systems | same-node signed benchmark, true checkpoint package size | Owner-independent perf witness | BLOCKED_ON_M4 |
| M8 | Scientific output | reproducibility bundle, figures, LaTeX technical report, negative results | Method/source audit | BLOCKED_ON_M2 |

## Small, independently parallelizable coding tasks (one writer per area)

**T01 (E0, docs):** resolve upstream repo commit and required versions; read paper v5 equations/Appendix D; record model license/dataset terms. *DoD:* pinned commit hashes and explicit legal/compatibility blockers, not a model install.

**T02 (E1, math):** independent pure numpy implementation of factorized forward and dense oracle. *DoD:* deterministic 3 seeds x 3 shapes x 3 spectra; signed U/V, h/g/ell and residual parity; tests identify wrong scale broadcasting.

**T03 (E1, storage):** theoretical bit accountant and binary sign pack/unpack. *DoD:* exact bit budget including FP16 scale vectors, padding, residual and metadata; measured bytes align with algebra, no implicit dense tensors in deployment serialization.

**T04 (E1, initializers):** isolated original Dual-SVID adapter and Joint-ITQ comparison. *DoD:* versioned upstream-compatibility table and reconstruction matrix; no copied noncommercial upstream code unless license/attribution reviewed. CPU implementation can be independent with credit.

**T05 (E0/E1, evidence):** manifest version, split lineage checker and read-only run summarizer. *DoD:* immutable manifests, schema validation, omission fails closed, no prompts/secrets, negative-action tests.

**T06 (E2 planning):** source-compatible Qwen3 layer map plus teacher/student memory envelope. *DoD:* no load/run until M3; GPU compatibility and owner/lease acceptance. Avoid new base-model support claims.

**T07 (E2 engineering):** separated QAT masks and objective parity. *DoD:* base/teacher frozen checks, master sign frozen checks, exact trainable parameter list and sealed checkpoints.

**T08 (E3):** independent heldout and tool-use probe harness with leakage witness. *DoD:* pinned corpus and task version, no tuning on final test, complete paired score table.

**T09 (E4):** antithetic scale-vector forward-only estimators with exact-gradient calibration. *DoD:* compare orthogonal/gaussian/no-update/wrong-sign under identical forward evaluation budget; report null results.

**T10 (E5):** same-device storage/latency benchmarks. *DoD:* prefill/decode separately, kernel listed, 10+ repetitions, no CUDA-result extrapolation to AMD.

**T11 (paper):** reproducibility report and arXiv-style independent draft. *DoD:* predictions before observations, uncertainty and negative outcomes, primary/upstream citations; avoid unsupported quantitative claims.

## Dependency graph

    T01 --> T04 --> M2 --> T06 --> M3 --> T07 --> M4 --> T08 --> M5 --> T09 --> M6
      \       \             /
       --> T02 --> M1 ------+
       --> T03 --> M1 ------+
       --> T05 --> all gates and T11
    M4 ---------------------> T10 --> M7
    M2, M5, M6, M7 ----------> T11 --> M8

M0/M1 can run in parallel on CPU. GPU allocation must not occur until M1, license and model/runtime gates close. Scientific publication can begin before any GPU experiment, but must not claim nonexistent results.

## Suggested execution coordination

- One "research decision" document owner; one writer per math/storage/manifest module; one independent reviewer of every gate. Record `task_id`, `parent_task`, `repo_head_sha`, `claimed_by`, `started_at`, `artifact_sha256`, `test_command`, `observed_outcome`, `reviewer` and `decision` in a portable audit record.
- An OpenCode subagent, if later used, may submit a draft PR and test report. It does **not** self-approve GPU, produce unverifiable quality claims, mutate main or deploy. Use distinct worktrees and explicit file ownership to prevent PR collision.
- Preserve original Dust PR #13 branches/results. This project's new branch starts at auto-finetune main, and its paper will use separate citation/reference namespaces.
- No unbounded continuous trainer, scheduler, GitHub self-hosted runner, implicit hosted-token fallback or automatic checkpoint promotion.

## Initial actionable review package

This documentation-only branch + draft PR is **M0 proposal, not a passed M0**. Next implementation PR should contain the E1 CPU oracle, bit budget tests and evidence-schema validator; its CI should use public synthetic inputs and never require torch, CUDA, provider calls or private file mounts. On review, lock initial hypotheses and authorize only a bounded E1 CPU smoke.

## Handoff checklist

1. Date/time and git SHA for documents and any code.
2. Explicit proposed hypothesis and unmodified null/control.
3. Which stage/gate was attempted, observed evidence with command plus hash.
4. Declared resource consumption and side effects.
5. Failed/incomplete checks and alternative explanations.
6. The smallest next reversible step; exact new approval required if outside current scope.
7. If no experiment was run, say **NOT_RUN** in the handoff rather than infer progress.

# Technical design: LittleBit-Fleet research platform

**2026-10-09 · proposal · no implementation claims**

## 1. Layers and training parameterizations

For original full-precision layer weight W in R^(d_out x d_in), represent the primary branch by sign factors U in {-1,+1}^(d_out x r) and V in {-1,+1}^(d_in x r) and scale vectors h (d_out), g (d_in), ell (r):

    W_hat_primary = diag(h) U diag(ell) V^T diag(g)
    forward(X)   = (((X * g) @ V) * ell) @ U^T * h

For a residual correction at a *fixed total bit budget*, represent W_hat = W_hat_primary + W_hat_residual with its own factors and scales. Residual is formed from W-W_hat_primary during initialization, then may be trained jointly. Keep the **same combined factor ranks/bit budget** in residual/no-residual comparisons. Direct dense reconstruction is an E1 test oracle, never the presumed fast inference implementation.

**Important:** Upstream full QAT learns continuous/master factor values behind SmoothSign during training. The novel scales-only arm freezes those master factors **and their signs**; trainable variables are limited to h,g,ell (and residual h/g/ell when enabled). This restriction changes the optimizer's function class. We must not label the result a reproduction of full LittleBit QAT.

### Reference baseline and experimental alternatives

| Variant | Factor initialization | Trainable variables | Purpose |
| --- | --- | --- | --- |
| B0 | FP16 teacher | none | quality and storage reference |
| B1 | original Dual-SVID | none | before-QAT loss |
| B2 | Joint-ITQ | none | better-initialization control |
| B3 | original Dual-SVID | master factors + scales | upstream full-QAT baseline |
| B4 | Joint-ITQ | master factors + scales | initialization x QAT interaction |
| B5 | Joint-ITQ | scales only | proposed efficient adaptation |
| B6 | Joint-ITQ | latent ell only | scale-subspace ablation |
| B7 | Joint-ITQ | scales only, forward-only estimator | new method, after gate |
| B8 | original + residual path | matched full/scales variants | quantify compensation contribution |
| B9 | compatible real Q4 checkpoint | none | practical inference alternative |

No model selection on the test set; checkpoint selection uses a *separate* validation split.

## 2. Effective bit accounting

For one primary branch at rank r, theoretical stored layer bits before padding/header:

    factor_bits = r * (d_out + d_in)            # sign U and sign V
    scale_bits  = b_scale * (d_out + d_in + r)  # typically b_scale=16
    layer_bits  = factor_bits + scale_bits

With residual branch at rank s, sum the two branches' factor and scale bits; add per-layer shape, metadata, padding, alignment, auxiliary quantization states and runtime-required tensors. If non-compressed biases/norms/embeddings/lm_head are present, add their actual stored bytes. Define two different denominators:

- `block_effective_bpw = compressed_transformer_linear_bits / original_transformer_linear_params`.
- `model_effective_bpw = total_deployed_model_artifact_bits / total_original_model_params`; report tokenizer/config and generic metadata **separately** as package overhead; include all model tensors. Also report `archive_bytes` including tokenizer/config and a reproducible map of exactly what was counted.

Check packed byte lengths, round-trip reconstructed output, index/padding and byte-order correctness; **no reliance on uncompressed PyTorch state_dict byte count as a compressed checkpoint**. Never equate BPW with training-time VRAM.

## 3. Component boundaries (planned)

    experiments/littlebit/
      cpu_oracle.py          # independently authored, no upstream code copied
      factor_init.py         # pinned adapter to vetted upstream Dual-SVID/ITQ or original implementation
      bit_budget.py          # exact formulas and serialization accounting
      matrix_fixtures.py     # generated seeds/shapes/spectra with hashes
      experiment_runner.py   # disarmed on import; explicit research CLI only
      qat_adapter.py         # isolated teacher/student QAT arm
      scale_params.py        # trainable mask and assertions
      forward_only.py        # antithetic estimation with explicit call-count budget
      eval_public.py         # fixed datasets and no leakage
      device_probe.py        # read-only GPU/software/owner inventory
      manifest.py            # schema validation, immutable structured evidence
      compare.py             # predeclared paired comparisons, negative result reporting
    research/littlebit-fleet/
      README.md
      PROJECT_CHARTER.md
      HYPOTHESES.md
      DESIGN.md
      EXPERIMENTS.md
      DELIVERY.md
      config/research-defaults.yaml
    papers/littlebit-fleet/
      main.tex
      references.bib

These files are a proposed future layout. **The existence of this design does not mean modules or experiments exist.**

## 4. Execution state machine

    DRAFT -> PREFLIGHTED -> APPROVED -> RUNNING -> EVIDENCED -> REVIEWED
       \        \          \         \          \          \
        HOLD <--- HOLD <----- HOLD <---- FAILED <-- FAILED     RELEASE_REVIEW

Transitions into APPROVED require explicit *human decision recorded in an immutable run manifest*. E0/E1 may be executed locally only after source and data verification. E2 and above require explicit GPU/time/resource authorization distinct from writing documents. A release-review status is not permission to deploy.

All backends are adapters that return measurements; none may call `src.scheduler`, `src.deploy`, `src.merge`, remote routers or shared fleet execution without a separately reviewed implementation and approval.

## 5. Source and data interfaces

Public preliminary data: synthetic matrix fixtures (E1), upstream-compatible *public* training examples (E2) and independent public heldout test tasks (E3). Candidate Qwen3-0.6B is proposed but not pinned or downloaded; model and tokenizer revision must be locked. For every dataset, store canonical path/version, dataset license, record count, content manifest hash, task-family grouping, document-source grouping, tokenizer version and exact split plan. Reject duplicates/near duplicates and shared document/task lineage across train, validation, test and calibration.

No private Opencode/Hermes transcripts or phone/NAS data. Manifests must contain hashes/counts and aggregate measures, never tokens, prompts, responses, API keys or client-sensitive traces.

## 6. Training and evaluation contracts

- Teacher checkpoint immutable, inference-only; student initial factor values immutable until explicitly permitted run. Record all trainable parameter masks and assert no base/teacher gradients where frozen.
- E2 starting objective follows original source: teacher/student KL output loss + 10 x intermediate hidden-state MSE, as implemented by upstream; compare behavior to true upstream code before claiming fidelity.
- Scales-only arm uses same objective, batches, tokens, stopping conditions and optimizer control except for trainable mask and its honestly reported parameter count.
- E4 forward-only method samples bounded antithetic directions and evaluates *same data* for +/- probes; use autograd only for local calibration control, never secretly as treatment. Count complete-model and cached-tail forwards separately. Reset RNG and baseline checkpoints between arms.
- Publish training/heldout CE, task success with confidence/dispersion, loss-trace shape, measured peak CPU RAM/VRAM and wall time, kernel selection, checkpoint sizes, energy where counters are credible.
- Performance controls: same host, device, prompt length, decode length, generation settings, cache state, warmup, 10+ measured repetitions, thermal/power logging, accelerator utilization and exact kernel; report prefill/decode separately.

## 7. Evidence manifest minimal logical schema

    run_id, preregistration_version, parent_experiment_id, experiment_arm
    timestamp_created_utc, prediction_recorded_utc, reviewed_by
    git_commit, upstream_git_commit, paper_version, license_review_status
    model_id, model_revision, tokenizer_hash, data_split_hashes
    source_config_sha256, dependency_lock_sha256, host_inventory_digest
    target_bpw, measured_block_bpw, measured_model_bpw
    resource_lease_id, max_runtime_seconds, max_gpu_memory_bytes
    training_enabled, deployment_enabled, provider_calls_enabled
    observations: {status, ce, ppl, task_metrics, elapsed, memory, artifact_hashes}
    audit: {split_checked, no_leakage, no_production_dispatch, files_written}
    exit_status, reviewer_decision

Treat all fields as typed, validated input. No `observations` are populated in preregistration manifests; if any required observed measurement is unavailable in a run result, mark **NOT_MEASURED** with a reason. One run -> one immutable JSON/JSONL record; a replacement produces a new run/revision rather than overwriting the old result.

## 8. Definition of ready for E2

E0 source/terms and env lock complete; E1 CPU oracle tests green; bit accounting equals serialization; clean dataset heldout separation witnessed; Qwen3 architecture/layer mapping matches upstream code; selected accelerator's GPU/VRAM, drivers and storage measured; consented time budget and lease in record; fault-injection/negative permission tests pass; and no concurrent training or storage pressure regression.

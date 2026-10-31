# LFM3-1.7B fleet inference experiments — 2026-10-30

Experimental, reproducible evidence; **not a fleet-wide speed ranking or production deployment**.

## Validated scope

* Checkpoint identity is bound by SHA-256. Q4_K_M: `3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4`; QAD-Q4_0: `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`.
* All benchmark numbers come from llama.cpp's own JSON `avg_ts`; all raw stdout, stderr, and command flags remain in receipts.
* **Single-request only.** Quantization variants, hardware, and token lengths differ. Results do not establish quality parity, user-facing latency or concurrency throughput.
* Paired `-pg` measurements report **combined** prompt+generation throughput; do not confuse them with decode-only speed.
* Includes failed and skipped tests. A failed runtime must not be counted as a model performance score.

## Reproduce on x1-370

```bash
python3 benchmark/lfm3-evidence/run.py --binary /home/scott/lfm3-research/llama-build/bin/llama-bench --ngl 99 --prompt 512 --decode 64 --repetitions 3
python3 benchmark/lfm3-evidence/run.py --binary /home/scott/lfm3-research/llama-build/bin/llama-bench --ngl 99 --prompt 8192 --decode 64 --repetitions 2 --paired --cache q8_0
```

## Reproduce on remote nodes (from trusted x1-370)

```bash
python3 benchmark/lfm3-evidence/fleet_runner.py --node deathstar-xps-8920 --prompt 512 --decode 64 --repetitions 3
python3 benchmark/lfm3-evidence/fleet_runner.py --node xwing --prompt 512 --decode 64 --repetitions 3
python3 benchmark/lfm3-evidence/summarize.py
```

The fleet runner requires SSH host-key verification, refuses unexpected model checksums, defaults to a 2 GiB available-memory safety gate, bounds test length, and runs the executable at low priority with a timeout. It does not change services.

## Known constraints

* Joyner was skipped when system RAM was below the safety threshold. No GPU claims for this node.
* xwing's earlier Vulkan binary did not load this LFM3 GGUF. The new Vulkan build requires an explicit Vulkan 1.3 shader-target workaround for its existing shaderc installation; it must still pass functional inference checks.
* x1-370 was observed with nearly full swap and a 97%-used system filesystem. Do not schedule unbounded concurrent jobs here.
* The main benchmarking runner should be used only against a known compatible llama.cpp version. Unsupported KV types and architectures must be recorded as failed, not silently bypassed.
* Future runs should test workload concurrency, p50/p95 latency, quality validation, GPU power and model-size-normalized memory before making deployment decisions.

## Verified extension results (same-date exploratory runs)

| Node | Backend | Checkpoint | Workload | Measurement |
| --- | --- | --- | --- | --- |
| x1-370 | Vulkan | Q4_K_M | 8192 prompt + 64 gen paired / FP16 KV | 1209.91 combined tok/s |
| x1-370 | Vulkan | Q4_K_M | 8192 prompt + 64 gen paired / Q8 KV | 1217.35 combined tok/s |
| xwing | CPU | QAD-Q4_0 | 512 prompt / 64 gen, separate | 132.19 prompt; 39.00 decode tok/s |
| xwing | Vulkan | QAD-Q4_0 | 512 prompt / 64 gen, separate | 3721.45 prompt; 152.16 decode tok/s |
| xwing | Vulkan | QAD-Q4_0 | 4096 prompt + 64 gen paired / FP16 KV | 2638.92 combined tok/s |
| xwing | Vulkan | QAD-Q4_0 | 4096 prompt + 64 gen paired / Q8 KV | 2599.33 combined tok/s |
| Deathstar | Vulkan | Q4_K_M | 512 prompt / 64 gen, separate | 787.49 prompt; 80.38 decode tok/s |

The 8192-token Q8-versus-FP16 result is within 1% and the 4096-token xwing result within 2%; **neither establishes a quantized-KV speed advantage**, and memory/quality measurements are not yet included. Avoid comparing combined throughput directly with separate prompt or decode throughput.

### xwing Vulkan build caveat

The bundled October 16 Vulkan binary rejected LFM3 GGUF. A fresh local build from the existing `llama-cpu-8ca4060-source` source snapshot passed using `-DGGML_VULKAN=ON`, Ninja, two build workers and a local `glslc` wrapper which substitutes `--target-env=vulkan1.3` for `--target-env=vulkan1.2`. The untracked source snapshot reported `build_commit=unknown`; verify source provenance separately before formal publication. **Do not apply the shader workaround globally.**

Future work: long-context output quality, actual KV-memory allocation, multiple concurrent clients with p95 latency, and ROCm HIP backend validation. These are not measured in this batch.

## 2026-10-30 extension: concurrency, DRM memory, retrieval probes

The localhost-only `concurrency.py` runner creates its own temporary `llama-server` at 127.0.0.1, warms it, issues short independent fixed-length requests, then terminates only its own server process. It does **not** measure production traffic or alter global services. A failed port preflight is retained as a failed receipt.

At x1-370 (Vulkan Q4_K_M, n_predict=48, three waves per level):

| KV type / run | 1 client | 2 clients | 4 clients |
| --- | --- | --- | --- |
| FP16, first run | 33.929 | 57.577 | 83.615 |
| Q8, first run | 42.621 | 78.529 | 118.360 |
| FP16, repeat | 51.622 | 80.186 | 112.200 |
| Q8, repeat | 52.767 | 67.572 | 104.212 |

Values are **aggregate generated tokens divided by full batch wall clock** (including prefill and scheduling), not isolated decode speed. Load and power were not controlled; the apparent KV performance ranking **reversed or narrowed on repetition**. Small-sample p95 values are descriptive only, not population latency guarantees.

`resource_profile.py` samples the child process RSS and systemwide AMD DRM VRAM/GTT counters during an 8192-prompt-plus-64-generation paired run. FP16 measured 970.58 combined tok/s, 422.27 MiB VRAM growth; Q8 measured 939.50 combined tok/s, 371.67 MiB growth. These are device-wide baseline deltas **not attributed KV allocations**. Multiple dedicated A/B repetitions are required.

`quality_probe.py` used deterministic synthetic inventory codes at three positions in a longer prompt. All three codes appeared in generated outputs with FP16 KV and Q8 KV, but neither achieved exact-answer formatting on all fixtures. Substring presence must **not** be reported as accuracy, safety, or output equivalence.

Run:

```bash
python3 benchmark/lfm3-evidence/concurrency.py --max-parallel 4 --repetitions 3 --tokens 48 --cache f16 --port 29620
python3 benchmark/lfm3-evidence/resource_profile.py --cache q8_0
python3 benchmark/lfm3-evidence/quality_probe.py --cache f16 --port 29618
python3 benchmark/lfm3-evidence/summarize_extended.py
python3 benchmark/lfm3-evidence/validate.py
```

These probes use fixed local paths and are intended for the controlled x1-370 lab only. All raw receipts are preserved; server logs are kept locally but intentionally omitted from GitHub evidence because they may contain prompt content. Future hardening should run through a per-device compatibility and exclusive GPU occupancy gate.

## ROCm gfx1151 compatibility smoke test

An isolated xwing ROCm HIP test compiled and executed a 4096-element SAXPY GPU kernel with 0 mismatches on the Radeon 8050S (gfx1151), using HIP 7.14 / clang 23. The exact source and executable return status are retained in `rocm_gfx1151_smoke.cpp` and `rocm-gfx1151-smoke-20261030.json`.

This confirms **HIP compiler + basic GPU kernel execution only**. It is **not an LFM3 native HIP inference benchmark**. The longer llama.cpp HIP build (with GPU target gfx1151 and flash-attention compilation disabled) was intentionally stopped at 111/267 incremental build steps to avoid leaving an unsupervised build running. The build cache remains on xwing; no production binaries were overwritten. x1-370 currently has an invalid HIP compiler path pointing into missing ROCm 7.2 files, and no HIP model benchmark was attempted there. The next ROCm milestone requires a complete source-provenance-checked llama.cpp HIP build and a verified model load.

Compile the smoke test on a gfx1151 node with a functioning ROCm installation:

```bash
hipcc -O2 --offload-arch=gfx1151 benchmark/lfm3-evidence/rocm_gfx1151_smoke.cpp -o /tmp/rocm-gfx1151-smoke
/tmp/rocm-gfx1151-smoke
```

## Retrieval scoring: require the final answer, not just a mentioned code

`score_retrieval.py` re-evaluates all three deterministic synthetic probes after the explicit `</think>` boundary, normalizes non-breaking hyphens and whitespace, and refuses to strip explanatory prose. On the recorded runs FP16 KV yielded **3/3 strict final answers**; Q8 KV yielded **1/3 strict final answers**, although the expected code occurred somewhere in all outputs. This tiny set is evidence of a potential **instruction-following format difference**, not a statistically valid quantization-quality conclusion. Raw synthetic generations are retained for audit.

```bash
python3 benchmark/lfm3-evidence/score_retrieval.py
```

## Additional hardware checks, October 30

**Deathstar RX 480 / Vulkan, Q4_K_M**, paired 4096-prompt + 64-generation workload with three repetitions: FP16 KV **617.77 combined tok/s**, Q8 KV **627.97 combined tok/s**. The roughly 1.7% difference is too small for a reliable conclusion; raw receipts include build and timing metadata.

**x1-370 gfx1150 ROCm packaging conflict:** the installed `hipcc` is **5.7.1-3**, while HIP runtime is **7.2.26015.70200** with ROCm LLVM clang-22. Legacy `hipcc` seeks a nonexistent clang-17 in the ROCm 7.2 tree; installed `libamdhip64-dev` also belongs to the 5.7 generation. The attempted direct compiler smoke test was unsuccessful. Neither ROCm installation nor system-level packages were changed. The full toolchain audit is retained locally on x1-370 as `rocm-x1-toolchain-20261030.json` and is not included in the published PR; only the high-level compatibility finding is published.

Do not infer HIP backend performance, or recommend replacing system packages without isolating service compatibility and ensuring rollback.

## Matched Vulkan/HIP benchmark harness (xwing)

`compare_backends.py` is a bounded, fail-closed A/B/B/A runner, requiring a compiled HIP executable. It runs `fleet_runner.py` sequentially with identical model SHA-256, batch sizes, threads, cache quantization, prompt/generation token counts, and three in-process repetitions; no benchmarks run concurrently. Output includes four underlying remote receipts and a summary of mean tokens/s per backend.

```bash
python3 benchmark/lfm3-evidence/compare_backends.py --prompt 512 --decode 64
python3 benchmark/lfm3-evidence/compare_backends.py --prompt 4096 --decode 64 --paired
```

A comparison is marked **failed** if the HIP binary is absent, the remote node is below the 4 GiB available-memory threshold, the model hash differs, the GPU backend is not the expected one, or any run fails. An A/B result alone is still exploratory: note that the HIP build config disables HIP flash attention, unlike the Vulkan configuration, and the node's background loads and clocks are not controlled. Do not report any speedup unless every source receipt is independently validated.

## Native ROCm inference is functional: measured Vulkan/HIP A/B/B/A results

On xwing's Radeon 8050S (gfx1151), a native HIP llama-bench binary built **successfully** from the extracted llama.cpp 8ca4060 source snapshot. It loads the LFM3-1.7B QAD-Q4_0 GGUF, offloads 99 requested GPU layers, and reports backend `ROCm`, not CPU fallback.

Two matched A/B/B/A experiments used the same SHA-256-bound checkpoint, four CPU threads, FP16 KV, no simultaneous bench processes, three internal repetitions per execution, and two executions per backend:

| Workload | Vulkan mean | HIP mean | Vulkan/HIP |
| --- | ---: | ---: | ---: |
| 512 prompt tokens, standalone prefill | 3583.54 tok/s | 2936.67 tok/s | 1.220x |
| 64 generated tokens, standalone decoding | 150.95 tok/s | 114.43 tok/s | 1.319x |
| Paired 4096 prompt + 64 generated | 2676.24 *combined* tok/s | 1763.00 *combined* tok/s | 1.518x |

Evidence: `backend-comparison-summary.json` links all four runs per experiment to their raw remote receipts. `backend-fingerprint-*.json` records both runtime binary SHA-256 hashes and matches six **selected** source files against the git tree at `8ca4060b7e30e76baaa5670ee1ec1b760b7f8552`. This is not a full source-tree attestation.

**Interpretation:** Vulkan won on this hardware and build configuration. The HIP CMake build set `GGML_CUDA_FA=OFF`, while Vulkan uses its separate, enabled attention kernels. Differences in kernel compilation, background loads and clocks remain uncontrolled. These results **do not** establish general Vulkan superiority, and must not be treated as equivalent-KV-quality evidence. HIP Flash Attention compatibility and repeatability under isolated load are the next optimization priorities.

Regenerate independently without GPUs:

```bash
python3 benchmark/lfm3-evidence/summarize_backends.py
python3 benchmark/lfm3-evidence/validate.py
```

The comparison summarizer retains failed A/B/B/A test records separately under `failed_comparisons`; it never assigns a speedup to an incomplete test. A failed raw bench or runner preflight remains available for follow-up and must not be silently omitted from evidence custody.

## Q8 KV regression reproduced on xwing HIP (4096+64 paired)

Using the same xwing QAD-Q4_0 model and matched ABBA harness, **Q8 KV** produced:

| Backend, two independent executions | Combined throughput |
| --- | ---: |
| Vulkan Q8 KV | 2686.80 and 2663.41 tok/s (mean 2675.11) |
| ROCm HIP Q8 KV | 216.39 and 219.41 tok/s (mean 217.90) |

Vulkan/HIP = **12.277x** for this paired 4096-prompt + 64-generation workload. Both HIP receipts reported backend `ROCm`, 99 requested GPU layers, model SHA-256 identity, and successful process completion. Per-run time for HIP was approximately 19 seconds versus 1.55 seconds on Vulkan.

For comparison, **FP16 KV** in the same ABBA protocol yielded Vulkan 2676.24 vs HIP 1763.00 *combined tok/s*. This suggests the HIP q8_0 K/V code path is disproportionately expensive in the current binary; the exact reason is **not verified**. HIP was built with `GGML_CUDA_FA=OFF`; attention kernels, VRAM residency, bandwidth, possible dequantization fallbacks, and device counters need targeted profiling. No production runtime defaults were changed. This evidence argues **against enabling HIP Q8 KV by default on xwing** before investigating.

Full files: `backend-compare-20261031T001313616787Z.json`, its four referenced `fleet-xwing-*.json` source receipts, and the deterministic `backend-comparison-summary.json`. All records remain reproducible and hash-linked to the same checkpoint.

## Q8 KV path isolation, October 30 follow-up

A second A/B/B/A test with Q8 KV, now separating 512-token prompt processing from 64-token generation, measured:

| Metric | Vulkan | HIP | Vulkan / HIP |
| --- | ---: | ---: | ---: |
| 512 prompt tokens | 3704.47 tok/s | 953.48 tok/s | 3.885x |
| 64 generated tokens | 150.46 tok/s | 95.67 tok/s | 1.573x |

This identifies a **larger prompt-processing penalty** than decoding penalty at short context, and an even more severe 4K combined-workload penalty. The causes are hypotheses, not proven by throughput measurements alone.

Flash Attention probes, captured in separate source receipts:

- Vulkan, Q8 K/V, 4096+64 paired, `-fa off`: **fails at context initialization** with `quantized V cache requires flash_attn to be enabled`; this unsupported experiment is recorded as `failed`.
- Native HIP binary compiled with `GGML_CUDA_FA=OFF`, Q8 K/V, 512+64 paired, `-fa on`: completes at 438.12 combined tok/s (single repetition).
- The same HIP binary, Q8 K/V, 4096+64 paired, `-fa on`: completes at 219.79 combined tok/s (single repetition), **not a meaningful improvement** over the 217.90 tok/s matched average with runtime FA set to auto.

These diagnostic runs are **not matched A/B comparisons**, and a runtime flag does not retroactively compile unavailable HIP Flash Attention kernels. The next optimization candidate is a *separate* HIP build compiled with `GGML_CUDA_FA=ON` and selected Q8/f16 attention kernels. Keep the original binaries and receipts unmodified. See tracking issue #105.

`fleet_runner.py` now accepts `--flash-attn auto|on|off` and records the chosen runtime flag in the raw receipt. Its default is still `auto`.

## Safe isolated candidate HIP Flash Attention build

The candidate binary is a **separate** gfx1151 HIP build tree, never a replacement for the successful no-FA baseline. `build_hip_fa_q8.sh` is a reproducible xwing-only helper with model-source hash checks (six selected files), RAM and disk preflight, exclusive lock, 2 low-priority compiler workers, and a 300-second hard timeout. It enables HIP Flash Attention while compiling only selected f16 and q8 K/V attention combinations, without touching system packages.

```bash
# Execute on xwing only, with the existing approved source tree:
bash benchmark/lfm3-evidence/build_hip_fa_q8.sh
```

An incomplete compile is **not evidence of successful FA support or speedup**. Verify the binary is present, that the exact HIP backend is loaded, and run a separately labelled, checksum-bound benchmark before comparing against any baseline.

## Candidate HIP Flash Attention measurement protocol

`compare_hip_fa.py` compares two **isolated native HIP binaries** (baseline no-FA vs candidate FA-on) on **xwing**, using a checksum-pinned QAD-Q4_0 model and the strict A/B/B/A order `HIP-noFA → HIP-FA → HIP-FA → HIP-noFA`. Both variants use identical runtime `-fa on`, Q8 K/V cache, 99 requested GPU layers and a 6 GiB remote available-memory admission gate. All four source receipts pin the **actual executable SHA-256**, verify the ROCm backend and record task timing and process outcomes. A candidate is not accepted merely because it compiles: we require model loading, valid outputs, and a separately executed quality test.

Only run this comparison after the candidate HIP FA binary is present. Both the original no-FA baseline and the existing Vulkan baseline remain unchanged.

```bash
python3 benchmark/lfm3-evidence/compare_hip_fa.py --prompt 512 --decode 64 --repetitions 3
python3 benchmark/lfm3-evidence/compare_hip_fa.py --prompt 4096 --decode 64 --repetitions 2
python3 benchmark/lfm3-evidence/summarize_hip_fa.py
python3 benchmark/lfm3-evidence/validate.py
```

The summary (once experiments exist) keeps *failed candidate comparisons* separate from completed measurements. Comparing two HIP variants does not by itself establish superiority over Vulkan; include a new matched Vulkan run for deployment recommendations. No modifications to production inference services are included.

## Confirmed native HIP Flash Attention recovery on gfx1151

The isolated candidate compiled with `GGML_CUDA_FA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF`, `GGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0`, and `CMAKE_HIP_ARCHITECTURES=gfx1151`. Its runtime, SHA-256 and selected source-file hashes are recorded in `hip-fa-build-20261031T004650045799Z.json`. The model SHA is `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`.

Two **complete** A/B/B/A experiments on xwing compared the native HIP baseline (no FA kernels compiled, but runtime `-fa on`) against the isolated FA-compiled candidate (also runtime `-fa on`), both using Q8 K/V and 99 GPU layers:

| Paired request, Q8 KV | HIP-noFA mean combined tok/s | HIP-FA mean combined tok/s | FA / noFA |
| --- | ---: | ---: | ---: |
| 512 prompt + 64 generation, 3 in-process repetitions | 463.54 | 760.21 | **1.640x** |
| 4096 prompt + 64 generation, 2 in-process repetitions | 210.86 | 2412.61 | **11.442x** |

Each mean is across two independent executions per binary, in A/B/B/A order. Every underlying receipt includes matching model SHA, separate **executable SHA-256**, ROCm backend identity, confirmed requested GPU layers, and the same runtime `-fa on` setting. Regenerate `hip-fa-summary.json` with `summarize_hip_fa.py`.

This substantially resolves the **measured** HIP Q8 regression on xwing, but does not conclusively identify the exact slow old-kernel dispatch. The source contains a possible quantized K/V fallback path, which was not observed as a warning in this baseline run. Background service loads and clocks were not standardized; results are exploratory.

**Release gates still open:** matched Vulkan-vs-FA A/B/B/A on the latest runtime, output correctness / instruction-following equivalence and long-duration p95 latency / memory validation. Do not replace the existing inference deployment solely on these speed measurements.

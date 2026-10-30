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

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
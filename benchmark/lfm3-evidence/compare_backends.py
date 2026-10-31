#!/usr/bin/env python3
"""Matched, non-concurrent Vulkan/HIP ABBA comparison on xwing.

Does not compile, change services, or accept unknown binaries/models. A missing
HIP backend produces a failed comparison receipt, never an invented speedup.
"""
import argparse
import datetime
import json
import pathlib
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_SHA = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"
RUNTIMES = {
    "Vulkan": "/home/scott/lfm3-research/build-vulkan-8ca4060/bin/llama-bench",
    "HIP": "/home/scott/lfm3-research/build-hip-gfx1151-bench/bin/llama-bench",
}
HIP_FA_RUNTIME = "/home/scott/lfm3-research/build-hip-gfx1151-fa-q8/bin/llama-bench"


def candidate_hashes():
    """Use immutable build-attestation and source fingerprint receipts."""
    original = sorted(ROOT.glob("backend-fingerprint-*.json"))
    candidate = sorted(ROOT.glob("hip-fa-build-*.json"))
    if len(original) != 1 or len(candidate) != 1:
        raise RuntimeError("missing or ambiguous binary attestation")
    old = json.loads(original[0].read_text())
    fa = json.loads(candidate[0].read_text())
    if old.get("status") != "completed" or fa.get("status") != "completed":
        raise RuntimeError("unverified candidate or Vulkan baseline")
    if fa.get("sha256", {}).get("model") != MODEL_SHA:
        raise RuntimeError("candidate model checkpoint mismatch")
    return {
        "Vulkan": old["runtime_binaries"]["Vulkan"]["binary_sha256"],
        "HIP": fa["sha256"]["hip_fa_q8"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=int, default=512, choices=[512, 4096])
    parser.add_argument("--decode", type=int, default=64, choices=[64])
    parser.add_argument("--repetitions", type=int, default=3, choices=[3])
    parser.add_argument("--cache", choices=["f16", "q8_0"], default="f16")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--hip-variant", choices=["no-fa", "fa-q8"], default="no-fa")
    opts = parser.parse_args()
    if opts.hip_variant == "fa-q8" and opts.cache != "q8_0":
        parser.error("FA candidate protocol currently supports Q8 KV only")
    runtimes = dict(RUNTIMES)
    binary_hashes = {}
    if opts.hip_variant == "fa-q8":
        runtimes["HIP"] = HIP_FA_RUNTIME
        binary_hashes = candidate_hashes()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / ("backend-compare-" + stamp + ".json")
    result = {
        "schema": "lfm3.xwing-matched-backends.v1",
        "created_utc": stamp,
        "model": "LFM3-1.7B-QAD-Q4_0.gguf",
        "expected_model_sha256": MODEL_SHA,
        "node": "xwing",
        "prompt": opts.prompt,
        "decode": opts.decode,
        "repetitions": opts.repetitions,
        "cache_type": opts.cache,
        "hip_variant": opts.hip_variant,
        "runtime_fa": "on" if opts.hip_variant == "fa-q8" else "auto",
        "binary_sha256": binary_hashes,
        "mode": "paired" if opts.paired else "separate",
        "order": ["Vulkan", "HIP", "HIP", "Vulkan"],
        "status": "not_run",
        "runs": [],
        "caveat": "Exploratory: different backend kernels; only 2 executions per backend; no temperature or background-load normalization.",
    }

    try:
        for backend in result["order"]:
            before = {p.name for p in ROOT.glob("fleet-xwing-*.json")}
            cmd = [
                sys.executable,
                str(ROOT / "fleet_runner.py"),
                "--node", "xwing",
                "--binary", runtimes[backend],
                "--gpu-layers", "99",
                "--model-name", result["model"],
                "--prompt", str(opts.prompt),
                "--decode", str(opts.decode),
                "--repetitions", str(opts.repetitions),
                "--cache", opts.cache,
                "--flash-attn", "on" if opts.hip_variant == "fa-q8" else "auto",
                "--timeout", "180",
                "--min-available-mib", "4096",
            ]
            if opts.paired:
                cmd.append("--paired")
            if binary_hashes:
                cmd += ["--expected-binary-sha256", binary_hashes[backend]]
            started = time.monotonic()
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=210)
            newer = sorted((p for p in ROOT.glob("fleet-xwing-*.json") if p.name not in before))
            if len(newer) != 1:
                raise RuntimeError(f"{backend}: expected exactly one new source receipt; got {len(newer)}")
            receipt = json.loads(newer[0].read_text())
            row = {
                "backend_expected": backend,
                "receipt": newer[0].name,
                "runner_exit_code": completed.returncode,
                "elapsed_s": round(time.monotonic() - started, 3),
                "status": receipt.get("status"),
                "backend_observed": receipt.get("backend"),
                "available_mib": receipt.get("available_mib"),
            }
            result["runs"].append(row)
            if completed.returncode or receipt.get("status") != "completed":
                raise RuntimeError(f"{backend}: benchmark failed or was safety-skipped; {receipt.get('reason')}")
            if receipt.get("observed_model_sha256") != MODEL_SHA:
                raise RuntimeError(f"{backend}: model checksum differs")
            if binary_hashes and receipt.get("observed_binary_sha256") != binary_hashes[backend]:
                raise RuntimeError(f"{backend}: executable SHA-256 mismatch")
            actual = [str(x).upper() for x in receipt.get("backend", [])]
            if not actual or not all(
                ("VULKAN" in name if backend == "Vulkan" else "HIP" in name or "ROCM" in name)
                for name in actual
            ):
                raise RuntimeError(f"{backend}: backend mismatch: {actual}")
            measurements = json.loads(receipt["stdout"])
            if not measurements:
                raise RuntimeError(f"{backend}: no measurements")
            for m in measurements:
                if not m.get("avg_ts") or m.get("n_prompt") not in (0, opts.prompt) or m.get("n_gen") not in (0, opts.decode):
                    raise RuntimeError(f"{backend}: incompatible measurement")
            row["measurements"] = [{
                "n_prompt": m["n_prompt"], "n_gen": m["n_gen"],
                "avg_ts": m["avg_ts"], "stddev_ts": m.get("stddev_ts"),
                "build_commit": m.get("build_commit"), "gpu_info": m.get("gpu_info"),
            } for m in measurements]
            time.sleep(1)

        aggregates = {}
        for backend in runtimes:
            runs = [r for r in result["runs"] if r["backend_expected"] == backend]
            keys = ["paired"] if opts.paired else ["prompt", "decode"]
            values = {
                "paired": lambda m: m["n_prompt"] > 0 and m["n_gen"] > 0,
                "prompt": lambda m: m["n_prompt"] > 0 and m["n_gen"] == 0,
                "decode": lambda m: m["n_prompt"] == 0 and m["n_gen"] > 0,
            }
            aggregates[backend] = {
                k: {
                    "mean_tps": statistics.mean(
                        next(m["avg_ts"] for m in r["measurements"] if values[k](m))
                        for r in runs
                    )
                } for k in keys
            }
        result["summary"] = aggregates
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = str(exc)
    finally:
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "path": str(path), "status": result["status"],
            "reason": result.get("reason"), "summary": result.get("summary"),
            "completed_runs": len(result["runs"]),
        }, indent=2))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

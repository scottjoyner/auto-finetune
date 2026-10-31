#!/usr/bin/env python3
"""Bounded, fail-closed A/B/B/A comparison of native HIP builds on xwing.

Baseline: same llama.cpp source snapshot built with GGML_CUDA_FA=OFF.
Candidate: isolated gfx1151 build compiled with GGML_CUDA_FA=ON.
Both execute Q8 KV with -fa on. Uses unique raw receipts; no production changes.
"""
import argparse
import datetime
import json
import pathlib
import shlex
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
ROOT_REMOTE = "/home/scott/lfm3-research"
BINARIES = {
    "HIP-noFA": ROOT_REMOTE + "/build-hip-gfx1151-bench/bin/llama-bench",
    "HIP-FA": ROOT_REMOTE + "/build-hip-gfx1151-fa-q8/bin/llama-bench",
}
CMAKE_CACHE = {
    "HIP-noFA": ROOT_REMOTE + "/build-hip-gfx1151-bench/CMakeCache.txt",
    "HIP-FA": ROOT_REMOTE + "/build-hip-gfx1151-fa-q8/CMakeCache.txt",
}
MODEL = "LFM3-1.7B-QAD-Q4_0.gguf"
MODEL_SHA = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"


def ssh(command, timeout=20):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=6", "xwing", command],
        capture_output=True, text=True, timeout=timeout,
    )


def prepare():
    hashes = {}
    for name, path in BINARIES.items():
        check = ssh("sha256sum " + shlex.quote(path) + "; grep -E "
                    + shlex.quote("^(GGML_CUDA_FA|GGML_HIP|GGML_CUDA_FA_QUANTS):")
                    + " " + shlex.quote(CMAKE_CACHE[name]))
        if check.returncode or not check.stdout.strip():
            raise RuntimeError(name + " executable or build cache missing")
        lines = check.stdout.strip().splitlines()
        hashes[name] = lines[0].split()[0]
        if len(hashes[name]) != 64:
            raise RuntimeError(name + " SHA invalid")
        flags = "\n".join(lines[1:])
        if "GGML_HIP:BOOL=ON" not in flags:
            raise RuntimeError(name + " is not built with HIP")
        if ("GGML_CUDA_FA:BOOL=ON" if name == "HIP-FA" else "GGML_CUDA_FA:BOOL=OFF") not in flags:
            raise RuntimeError(name + " has incorrect Flash Attention build setting")
        if name == "HIP-FA" and "q8_0-q8_0" not in flags:
            raise RuntimeError("candidate Q8 kernel not configured")
    if hashes["HIP-noFA"] == hashes["HIP-FA"]:
        raise RuntimeError("candidate binary is the same as baseline")
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=int, choices=[512, 4096], default=512)
    parser.add_argument("--decode", type=int, choices=[64], default=64)
    parser.add_argument("--repetitions", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--timeout", type=int, choices=[120, 180], default=180)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / ("hip-fa-compare-" + stamp + ".json")
    experiment = {
        "schema": "lfm3.hip-fa-abba.v1", "created_utc": stamp, "node": "xwing",
        "model": MODEL, "expected_model_sha256": MODEL_SHA,
        "cache": "q8_0", "flash_attention_runtime": "on",
        "prompt": args.prompt, "generation": args.decode,
        "repetitions": args.repetitions,
        "order": ["HIP-noFA", "HIP-FA", "HIP-FA", "HIP-noFA"],
        "status": "not_run", "runs": [],
        "caveats": "Small sequential sample; background device load, clocks and memory bandwidth not controlled.",
    }
    try:
        hashes = prepare()
        experiment["runtime_binary_sha256"] = hashes
        for name in experiment["order"]:
            before = {p.name for p in ROOT.glob("fleet-xwing-*.json")}
            command = [
                sys.executable, str(ROOT / "fleet_runner.py"),
                "--node", "xwing", "--binary", BINARIES[name],
                "--model-name", MODEL, "--gpu-layers", "99",
                "--min-available-mib", "6144", "--cache", "q8_0",
                "--flash-attn", "on", "--paired",
                "--prompt", str(args.prompt), "--decode", str(args.decode),
                "--repetitions", str(args.repetitions),
                "--timeout", str(args.timeout),
                "--expected-binary-sha256", hashes[name],
            ]
            start = time.monotonic()
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout + 35)
            newest = sorted(p for p in ROOT.glob("fleet-xwing-*.json") if p.name not in before)
            if len(newest) != 1:
                raise RuntimeError(f"{name}: expected exactly one receipt; found {len(newest)}")
            source = json.loads(newest[0].read_text())
            row = {
                "expected_variant": name,
                "receipt": newest[0].name,
                "runner_exitcode": process.returncode,
                "status": source.get("status"),
                "elapsed_wall_seconds": round(time.monotonic() - start, 3),
            }
            experiment["runs"].append(row)
            if process.returncode or source.get("status") != "completed":
                raise RuntimeError(f"{name}: benchmark failed; {source.get('reason')}")
            if (source.get("observed_model_sha256") != MODEL_SHA or
                    source.get("observed_binary_sha256") != hashes[name]):
                raise RuntimeError(f"{name}: hash mismatch")
            if source.get("backend") != ["ROCm"]:
                raise RuntimeError(f"{name}: wrong backend {source.get('backend')}")
            if source.get("flash_attn") != "on" or source.get("cache_type") != "q8_0":
                raise RuntimeError(f"{name}: incompatible attention or cache flags")
            measurements = json.loads(source["stdout"])
            if len(measurements) != 1 or (
                    measurements[0]["n_prompt"] != args.prompt or
                    measurements[0]["n_gen"] != args.decode):
                raise RuntimeError(f"{name}: incorrect prompt+generation test")
            if measurements[0]["avg_ts"] <= 0:
                raise RuntimeError(f"{name}: invalid throughput")
            row["avg_ts"] = measurements[0]["avg_ts"]
            row["stddev_ts"] = measurements[0].get("stddev_ts")
            row["gpu_info"] = measurements[0].get("gpu_info")
            time.sleep(1)
        means = {
            name: statistics.mean(row["avg_ts"] for row in experiment["runs"] if row["expected_variant"] == name)
            for name in BINARIES
        }
        experiment["summary"] = {
            "mean_combined_tps": means,
            "candidate_divided_by_baseline": means["HIP-FA"] / means["HIP-noFA"],
        }
        experiment["status"] = "completed"
    except Exception as error:
        experiment.update(status="failed", reason=str(error))
    finally:
        path.write_text(json.dumps(experiment, indent=2) + "\n")
        print(json.dumps({
            "receipt": str(path),
            "status": experiment["status"],
            "reason": experiment.get("reason"),
            "summary": experiment.get("summary"),
            "completed_runs": len(experiment["runs"]),
        }, indent=2))
    return 0 if experiment["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

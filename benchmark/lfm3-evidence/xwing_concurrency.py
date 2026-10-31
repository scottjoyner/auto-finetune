#!/usr/bin/env python3
"""Bounded localhost-only concurrency benchmark for isolated xwing FA llama-server.

No external requests, configuration changes or production service interactions.
Kills only its own subprocess process group. Synthetic prompts, no user data.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path("/home/scott/lfm3-research/concurrency-receipts")
MODEL = pathlib.Path("/home/scott/lfm3-research/gguf/LFM3-1.7B-QAD-Q4_0.gguf")
MODEL_SHA = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"
BIN = pathlib.Path("/home/scott/lfm3-research/build-hip-gfx1151-fa-q8/bin/llama-server")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request(url: str, body: dict | None = None, timeout: int = 10) -> dict:
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=timeout) as stream:
        response = stream.read(1_000_000)
    result = json.loads(response)
    if not isinstance(result, dict):
        raise RuntimeError("Unexpected server response")
    return result


def nearest_rank(values: list[float], percentile: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute percentile over empty population")
    import math
    return values[max(0, min(len(values) - 1, math.ceil(percentile * len(values)) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29644)
    parser.add_argument("--waves", type=int, choices=[2, 3, 4], default=3)
    parser.add_argument("--max-parallel", type=int, choices=[2, 4], default=4)
    parser.add_argument("--tokens", type=int, choices=[32, 48, 64], default=48)
    parser.add_argument("--cache", choices=["q8_0"], default="q8_0")
    args = parser.parse_args()
    if not (20000 <= args.port <= 65000):
        parser.error("port outside isolated test range")
    ROOT.mkdir(exist_ok=True, parents=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / ("xwing-concurrency-" + stamp + ".json")
    log_path = ROOT / ("xwing-concurrency-" + stamp + ".log")
    record = {
        "schema": "lfm3.xwing-hip-fa-concurrency.v1",
        "timestamp_utc": stamp, "node": "xwing", "status": "not_run",
        "model": str(MODEL), "model_sha256_expected": MODEL_SHA,
        "server_binary": str(BIN),
        "cache_type": args.cache, "flash_attn": "on",
        "max_parallel": args.max_parallel, "waves_per_load": args.waves,
        "generated_token_cap": args.tokens,
        "results": [],
        "caveats": ["Localhost synthetic requests only; no production requests",
                    "Client-side full-response latencies include prompt, queue and generation",
                    "Small-sample p95 is descriptive, not a stable percentile estimate"],
    }
    proc = None
    logfile = None
    try:
        if socket.gethostname().split(".")[0] != "xwing":
            raise RuntimeError("This runner requires xwing")
        mem = pathlib.Path("/proc/meminfo").read_text()
        available = int(next(x.split()[1] for x in mem.splitlines()
                             if x.startswith("MemAvailable:"))) // 1024
        stat = os.statvfs(str(ROOT))
        free_disk_mib = stat.f_bavail * stat.f_frsize // (1024 * 1024)
        record["preflight"] = {"available_mib": available, "disk_free_mib": free_disk_mib}
        if available < 6144 or free_disk_mib < 32768:
            record["status"] = "skipped"
            record["reason"] = "memory or disk safety threshold"
            return 0
        if not MODEL.is_file() or not BIN.is_file():
            raise RuntimeError("Model or isolated server binary missing")
        observed = sha256(MODEL)
        record["model_sha256_observed"] = observed
        if observed != MODEL_SHA:
            raise RuntimeError("Model hash changed")
        executable = sha256(BIN)
        record["server_binary_sha256"] = executable
        if len(executable) != 64:
            raise RuntimeError("Executable hash missing")

        # Reserve/release port immediately before launch; fail if in use.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        server_cmd = [
            "nice", "-n", "12", str(BIN), "-m", str(MODEL),
            "--host", "127.0.0.1", "--port", str(args.port),
            "--no-webui", "--log-disable", "-ngl", "99", "-t", "4",
            "-c", "8192", "-np", str(args.max_parallel),
            "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
        ]
        record["server_command"] = server_cmd
        logfile = log_path.open("w")
        proc = subprocess.Popen(server_cmd, stdout=logfile, stderr=subprocess.STDOUT,
                                start_new_session=True)
        record["server_pid"] = proc.pid
        base_url = f"http://127.0.0.1:{args.port}"
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {proc.returncode}")
            try:
                health = request(base_url + "/health", timeout=2)
                if health.get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(.5)
        else:
            raise RuntimeError("Server did not become healthy within 75s")
        record["server_ready"] = True

        def task(n: int) -> dict:
            prompt = ("You are a technical assistant. Explain why reproducible inference "
                      "benchmarks require model fingerprints and latency measurements. "
                      "Respond in two concise sentences. ") * 2 + f"Case {n}."
            body = {"prompt": prompt, "n_predict": args.tokens,
                    "temperature": 0, "seed": 1000 + n,
                    "stream": False, "cache_prompt": False}
            start = time.monotonic()
            response = request(base_url + "/completion", body, timeout=95)
            time_s = time.monotonic() - start
            timing = response.get("timings", {})
            predicted = timing.get("predicted_n")
            if response.get("error") or not isinstance(predicted, int) or predicted <= 0:
                raise RuntimeError("Completion returned invalid token count")
            return {"elapsed_seconds": round(time_s, 4), "predicted_tokens": predicted,
                    "prompt_tokens": timing.get("prompt_n"),
                    "model_generation_ms": timing.get("predicted_ms"),
                    "content_length": len(response.get("content", "")),
                    "stop": response.get("stop")}

        record["warmup"] = task(999)
        for n in [1, 2, 4]:
            if n > args.max_parallel:
                continue
            waves = []
            for wave in range(args.waves):
                start = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                    tasks = [pool.submit(task, wave * 100 + j) for j in range(n)]
                    outcomes = [f.result(timeout=100) for f in tasks]
                wall_s = time.monotonic() - start
                waves.append({"wall_seconds": round(wall_s, 4),
                              "requests": outcomes,
                              "aggregate_generated_tps": round(
                                  sum(x["predicted_tokens"] for x in outcomes) / wall_s, 3)})
            latencies = [task["elapsed_seconds"] for w in waves for task in w["requests"]]
            record["results"].append({
                "concurrency": n, "waves": waves, "samples": len(latencies),
                "median_client_latency_s": round(statistics.median(latencies), 4),
                "sample_p95_client_latency_s": round(nearest_rank(latencies, .95), 4),
                "mean_aggregate_generated_tps": round(statistics.mean(
                    w["aggregate_generated_tps"] for w in waves), 3),
            })
        record["status"] = "completed"
    except Exception as error:
        record["status"] = "failed"
        record["reason"] = str(error)
    finally:
        if proc:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=12)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if logfile:
            logfile.close()
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({
            "receipt": str(path), "status": record["status"],
            "reason": record.get("reason"),
            "model_sha256": record.get("model_sha256_observed"),
            "server_sha256": record.get("server_binary_sha256"),
            "results": [{k: v for k, v in row.items() if k != "waves"}
                        for row in record["results"]],
        }, indent=2))
    return 0 if record["status"] in ("completed", "skipped") else 2


if __name__ == "__main__":
    sys.exit(main())

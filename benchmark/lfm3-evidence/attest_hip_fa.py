#!/usr/bin/env python3
"""Verify FA-on HIP executable and selected pinned source files on xwing."""
import datetime
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MODEL = "/home/scott/lfm3-research/gguf/LFM3-1.7B-QAD-Q4_0.gguf"
BASE = "/home/scott/lfm3-research/build-hip-gfx1151-bench"
CANDIDATE = "/home/scott/lfm3-research/build-hip-gfx1151-fa-q8"
SOURCE = "/home/scott/lfm3-research/llama-cpu-8ca4060-source"
PIN = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                           "-o", "ConnectTimeout=6", "xwing", cmd],
                          capture_output=True, text=True, timeout=40)


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / ("hip-fa-build-" + stamp + ".json")
    data = {"schema": "lfm3.hip-fa-build-provenance.v1",
            "created_utc": stamp, "status": "not_run", "model_sha256": PIN}
    try:
        original = json.loads(next(ROOT.glob("backend-fingerprint-*.json")).read_text())
        require(original.get("status") == "completed", "baseline source fingerprint missing")
        data["source_commit_reference"] = original["source_commit"]
        sources = {}
        for item in original["source_files"]:
            rel = item["path"]
            output = ssh("sha256sum " + shlex.quote(SOURCE + "/" + rel))
            require(output.returncode == 0, "snapshot source missing " + rel)
            digest = output.stdout.split()[0]
            require(digest == item["git_tree_file_sha256"], "source mismatch " + rel)
            sources[rel] = digest
        data["selected_source_sha256"] = sources
        digest_paths = [MODEL, BASE + "/bin/llama-bench",
                        CANDIDATE + "/bin/llama-bench"]
        checks = ssh("sha256sum " + " ".join(shlex.quote(x) for x in digest_paths))
        require(checks.returncode == 0, "candidate or baseline executable unavailable")
        hashes = [x.split()[0] for x in checks.stdout.splitlines()]
        require(len(hashes) == len(digest_paths) and hashes[0] == PIN, "model mismatch")
        require(hashes[1] != hashes[2], "candidate executable equals baseline")
        data["sha256"] = dict(zip(["model", "hip_no_fa", "hip_fa_q8"], hashes))
        cache_path = CANDIDATE + "/CMakeCache.txt"
        flags = ssh("grep -E '^(GGML_HIP|GGML_CUDA_FA|GGML_CUDA_FA_QUANTS|CMAKE_HIP_ARCHITECTURES):' " +
                    shlex.quote(cache_path))
        require(flags.returncode == 0, "missing HIP CMake flags")
        data["candidate_cmake_flags"] = flags.stdout.strip().splitlines()
        require(any("GGML_HIP:BOOL=ON" in s for s in data["candidate_cmake_flags"]), "HIP not on")
        require(any("GGML_CUDA_FA:BOOL=ON" in s for s in data["candidate_cmake_flags"]), "FA not on")
        require(any("q8_0-q8_0" in s for s in data["candidate_cmake_flags"]), "Q8 FA kernels not selected")
        version = ssh("env LD_LIBRARY_PATH=" + shlex.quote(CANDIDATE + "/bin") +
                      " " + shlex.quote(CANDIDATE + "/bin/llama-bench") + " --version")
        require(version.returncode == 0, "candidate --version failed")
        data["runtime_version_output"] = (version.stdout + version.stderr)[-5000:]
        require("ROCm" in data["runtime_version_output"] and "gfx1151" in data["runtime_version_output"],
                "candidate does not enumerate ROCm gfx1151")
        data["status"] = "completed"
    except Exception as exc:
        data["status"] = "failed"
        data["reason"] = str(exc)
    finally:
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps({"receipt": str(path), "status": data["status"],
                          "reason": data.get("reason"), "sha256": data.get("sha256"),
                          "source_files_checked": len(data.get("selected_source_sha256", {}))}, indent=2))
    return 0 if data["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Read-only, source-to-runtime hardware fingerprint for xwing benchmark binaries."""
import datetime
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SOURCE_REPO = "/home/scott/lfm3-research/llama.cpp"
SOURCE_SNAPSHOT = "/home/scott/lfm3-research/llama-cpu-8ca4060-source"
FILES = (
    "CMakeLists.txt",
    "src/llama-model.cpp",
    "src/llama-context.cpp",
    "ggml/src/ggml-hip/CMakeLists.txt",
    "ggml/src/ggml-vulkan/CMakeLists.txt",
    "tools/llama-bench/llama-bench.cpp",
)
BUILD_ROOT = "/home/scott/lfm3-research"
RUNTIMES = {
    "Vulkan": BUILD_ROOT + "/build-vulkan-8ca4060/bin/llama-bench",
    "HIP": BUILD_ROOT + "/build-hip-gfx1151-bench/bin/llama-bench",
}


def call(command, timeout=30):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def ssh(command):
    return call(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                 "-o", "StrictHostKeyChecking=yes", "xwing", command])


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    r = {"schema": "lfm3.xwing-backend-fingerprint.v1",
         "created_utc": stamp, "status": "not_run",
         "source_reference": "git HEAD on x1-370",
         "source_check_scope": "six selected files, NOT a complete source tree attestation",
         "source_files": [], "runtime_binaries": {}}
    path = ROOT / ("backend-fingerprint-" + stamp + ".json")
    try:
        head = call(["git", "-C", SOURCE_REPO, "rev-parse", "HEAD"])
        if head.returncode:
            raise RuntimeError("source repo unavailable")
        r["source_commit"] = head.stdout.strip()
        for rel in FILES:
            tracked = subprocess.run(
                ["git", "-C", SOURCE_REPO, "show", "HEAD:" + rel],
                capture_output=True, timeout=20
            )
            if tracked.returncode:
                raise RuntimeError("missing tracked source file " + rel)
            expected = hashlib.sha256(tracked.stdout).hexdigest()
            remote = ssh("sha256sum " + shlex.quote(SOURCE_SNAPSHOT + "/" + rel))
            if remote.returncode:
                raise RuntimeError("missing xwing source " + rel)
            actual = remote.stdout.split()[0]
            record = {"path": rel, "git_tree_file_sha256": expected,
                      "xwing_snapshot_sha256": actual, "match": expected == actual}
            r["source_files"].append(record)
            if not record["match"]:
                raise RuntimeError("source snapshot mismatch " + rel)
        for backend, binary in RUNTIMES.items():
            result = ssh("sha256sum " + shlex.quote(binary))
            if result.returncode:
                raise RuntimeError("missing runtime binary " + backend)
            cache = BUILD_ROOT + ("/build-hip-gfx1151-bench/CMakeCache.txt" if backend == "HIP"
                                  else "/build-vulkan-8ca4060/CMakeCache.txt")
            flags = ssh("grep -E '^(GGML_HIP|GGML_VULKAN|GGML_CUDA_FA|CMAKE_HIP_ARCHITECTURES):' "
                        + shlex.quote(cache))
            r["runtime_binaries"][backend] = {
                "path": binary, "binary_sha256": result.stdout.split()[0],
                "cmake_flags": flags.stdout.strip().splitlines(),
            }
        hip = ssh("hipcc --version | head -3")
        if hip.returncode:
            raise RuntimeError("HIP compiler version unavailable")
        r["hipcc_version"] = hip.stdout.strip().splitlines()
        uname = ssh("uname -r")
        r["kernel"] = uname.stdout.strip()
        r["status"] = "completed"
    except Exception as exc:
        r["status"] = "failed"
        r["reason"] = str(exc)
    finally:
        path.write_text(json.dumps(r, indent=2) + "\n")
        print(json.dumps({"path": str(path), "status": r["status"],
                          "source_commit": r.get("source_commit"),
                          "source_files_match": len([x for x in r["source_files"] if x["match"]]),
                          "runtime_binaries": {k: v["binary_sha256"] for k, v in
                                               r["runtime_binaries"].items()},
                          "reason": r.get("reason")}, indent=2))
    return 0 if r["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

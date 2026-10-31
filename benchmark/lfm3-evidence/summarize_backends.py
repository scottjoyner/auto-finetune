#!/usr/bin/env python3
"""Regenerate quantitative xwing Vulkan/HIP comparisons from raw ABBA receipts."""
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def fail(reason):
    raise ValueError(reason)


def main():
    paths = sorted(ROOT.glob("backend-compare-*.json"))
    sources = sorted(ROOT.glob("backend-fingerprint-*.json"))
    if not paths or len(sources) != 1:
        fail("need comparison receipts and exactly one source fingerprint")
    fp = json.loads(sources[0].read_text())
    if fp.get("status") != "completed" or len(fp.get("source_files", [])) != 6:
        fail("incomplete source fingerprint")
    if not all(item.get("match") for item in fp["source_files"]):
        fail("source hash mismatch")
    summaries = []
    for path in paths:
        r = json.loads(path.read_text())
        if r.get("status") != "completed":
            fail("incomplete comparison " + path.name)
        if r.get("order") != ["Vulkan", "HIP", "HIP", "Vulkan"]:
            fail("ABBA order mismatch " + path.name)
        if len(r.get("runs", [])) != 4:
            fail("incomplete comparison run count " + path.name)
        for run in r["runs"]:
            source_path = ROOT / run["receipt"]
            if not source_path.is_file() or source_path.name != run["receipt"]:
                fail("missing raw receipt " + run["receipt"])
            source = json.loads(source_path.read_text())
            if source.get("observed_model_sha256") != r.get("expected_model_sha256"):
                fail("incompatible model hashes")
            if source.get("status") != "completed":
                fail("raw receipt not completed")
        measurements = []
        if r["mode"] == "paired":
            metrics = {"paired_combined_tps": "paired"}
        else:
            metrics = {"prompt_tps": "prompt", "decode_tps": "decode"}
        for metric, key in metrics.items():
            vk = r["summary"]["Vulkan"][key]["mean_tps"]
            hip = r["summary"]["HIP"][key]["mean_tps"]
            if vk <= 0 or hip <= 0:
                fail("nonpositive throughput")
            measurements.append({
                "metric": metric,
                "vulkan_tps": round(vk, 3),
                "hip_tps": round(hip, 3),
                "vulkan_divided_by_hip": round(vk / hip, 4),
                "hip_relative_to_vulkan_percent": round(100 * hip / vk, 2),
            })
        summaries.append({
            "comparison_receipt": path.name,
            "model": r["model"],
            "model_sha256": r["expected_model_sha256"],
            "prompt_tokens": r["prompt"],
            "generation_tokens": r["decode"],
            "cache_type": r["cache_type"],
            "repetitions_per_run": r["repetitions"],
            "executions_per_backend": 2,
            "source_receipts": [x["receipt"] for x in r["runs"]],
            "metrics": measurements,
        })
    output = {
        "schema": "lfm3.xwing-backend-comparison-summary.v1",
        "source_fingerprint_receipt": sources[0].name,
        "source_git_commit": fp["source_commit"],
        "hardware": "xwing, AMD Radeon 8050S, gfx1151",
        "model": "LFM3-1.7B-QAD-Q4_0",
        "tested_cache": "f16",
        "caveats": [
            "HIP built with GGML_CUDA_FA=OFF; Vulkan has different attention kernels.",
            "Two backend executions, three llama-bench repetitions per execution.",
            "Node production services and hardware clocks not isolated or normalized.",
            "Paired combined tokens/s must not be read as standalone decode speed.",
            "No broad model accuracy or correctness equivalence verified.",
            "Only six source snapshot files matched a pinned git commit, not a complete tree attestation.",
        ],
        "comparisons": sorted(summaries, key=lambda x: x["prompt_tokens"]),
    }
    dest = ROOT / "backend-comparison-summary.json"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    for comp in output["comparisons"]:
        for m in comp["metrics"]:
            print(comp["prompt_tokens"], m["metric"], m["vulkan_tps"],
                  m["hip_tps"], "Vulkan/HIP", m["vulkan_divided_by_hip"])
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print("FAIL", str(exc), file=sys.stderr)
        sys.exit(1)

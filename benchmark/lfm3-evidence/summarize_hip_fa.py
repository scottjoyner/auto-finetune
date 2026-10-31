#!/usr/bin/env python3
"""Rebuild HIP Flash Attention candidate comparison summary from raw receipts."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT = ROOT / "hip-fa-summary.json"
ORDER = ["HIP-noFA", "HIP-FA", "HIP-FA", "HIP-noFA"]


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    inputs = sorted(ROOT.glob("hip-fa-compare-*.json"))
    if not inputs:
        require(not OUTPUT.exists(), "orphan HIP FA summary without any source comparison")
        print("No HIP FA comparisons recorded")
        return 0
    results = []
    failures = []
    for path in inputs:
        record = json.loads(path.read_text())
        require(record.get("schema") == "lfm3.hip-fa-abba.v1", path.name + ": bad schema")
        require(record.get("order") == ORDER, path.name + ": bad sequence")
        require(record.get("cache") == "q8_0", path.name + ": unexpected cache")
        require(record.get("flash_attention_runtime") == "on", path.name + ": unexpected runtime flag")
        runs = record.get("runs", [])
        require(len(runs) <= 4, path.name + ": too many runs")
        for index, run in enumerate(runs):
            require(run.get("expected_variant") == ORDER[index], path.name + ": wrong sequence")
            source_path = ROOT / run.get("receipt", "")
            require(source_path.is_file() and source_path.name.startswith("fleet-xwing-"), path.name + ": missing source")
            src = json.loads(source_path.read_text())
            require(src.get("status") == run.get("status"), path.name + ": source status mismatch")
            require(src.get("observed_model_sha256") == record.get("expected_model_sha256"),
                    path.name + ": source model mismatch")
            if src.get("status") == "completed":
                require(src.get("backend") == ["ROCm"], path.name + ": wrong backend")
                require(src.get("flash_attn") == "on" and src.get("cache_type") == "q8_0",
                        path.name + ": wrong attention mode")
                expect_bin = record.get("runtime_binary_sha256", {}).get(run["expected_variant"])
                require(expect_bin and src.get("observed_binary_sha256") == expect_bin,
                        path.name + ": wrong runtime binary")
        if record.get("status") != "completed":
            failures.append({"receipt": path.name, "status": record.get("status"),
                             "reason": record.get("reason", "no reason")})
            continue
        require(len(runs) == 4 and all(r.get("status") == "completed" for r in runs),
                path.name + ": incomplete successful ABBA")
        measured = record.get("summary", {}).get("mean_combined_tps", {})
        old = measured.get("HIP-noFA", 0)
        new = measured.get("HIP-FA", 0)
        require(old > 0 and new > 0, path.name + ": nonpositive throughput")
        results.append({
            "comparison_receipt": path.name, "prompt_tokens": record["prompt"],
            "generation_tokens": record["generation"], "cache": record["cache"],
            "baseline_combined_tps": round(old, 3),
            "candidate_combined_tps": round(new, 3),
            "candidate_divided_by_baseline": round(new / old, 4),
            "sha256_baseline": record["runtime_binary_sha256"]["HIP-noFA"],
            "sha256_candidate": record["runtime_binary_sha256"]["HIP-FA"],
            "source_receipts": [x["receipt"] for x in runs],
            "repetitions_per_execution": record["repetitions"],
        })
    summary = {
        "schema": "lfm3.hip-fa-summary.v1",
        "warnings": ["No production cutover or tuning applied",
                     "Four sequential runs per comparison; not clock/temperature controlled",
                     "Candidate build flag alone does not establish kernel selection",
                     "llama-bench timings do not establish output equivalence"],
        "comparisons": results,
        "failed_comparisons": failures,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2) + "\n")
    for x in results:
        print(x["prompt_tokens"], "candidate / no-FA", x["candidate_divided_by_baseline"])
    print("saved", OUTPUT, len(results), "completed,", len(failures), "failed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print("FAIL", str(exc), file=sys.stderr)
        sys.exit(1)

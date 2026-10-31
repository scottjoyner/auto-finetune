#!/usr/bin/env python3
"""Deterministic synthetic output agreement: native HIP no-FA vs FA-on candidate.

The same tiny locally compiled llama.cpp simple harness is dynamically linked
against two mutually isolated HIP runtime directories. The model's embedded
chat template is rendered with Jinja2, with generation-span annotations removed
(they have no textual effect). This is a smoke test, NOT broad model accuracy.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

from jinja2 import Environment

ROOT = pathlib.Path(__file__).resolve().parent
REMOTE = "/home/scott/lfm3-research"
MODEL = REMOTE + "/gguf/LFM3-1.7B-QAD-Q4_0.gguf"
MODEL_HASH = "43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452"
CLI = REMOTE + "/lfm3-quality-simple"
TEMPLATE = REMOTE + "/lfm3-chat-template.j2"
BUILDS = {
    "HIP-noFA": REMOTE + "/build-hip-gfx1151-bench/bin",
    "HIP-FA": REMOTE + "/build-hip-gfx1151-fa-q8/bin",
}
FIXTURES = [
    {"id": "arithmetic-17x19", "prompt": "What is 17 multiplied by 19? Answer only the numeral.", "expected": "323"},
    {"id": "token-ready", "prompt": "Reply with exactly the uppercase word READY.", "expected": "READY"},
    {"id": "inventory-code", "prompt": "Inventory row 14 has verification code ZN6P-4021. What code belongs to row 14? Respond only with the code.", "expected": "ZN6P-4021"},
    {"id": "sequence", "prompt": "What number comes next in the sequence 4, 8, 16, 32? Answer only the number.", "expected": "64"},
    {"id": "literal", "prompt": "Return exactly the string BETA-17 with no explanation.", "expected": "BETA-17"},
]
long_rows = [
    f"Inventory row {i:03d}: component classification standard, inspection complete, custody status stable."
    for i in range(90)
]
long_rows[43] = "Inventory row 043: verification code VQ8P-6142, inspection complete, custody status stable."
FIXTURES.append({
    "id": "long-context-retrieval",
    "prompt": "Review the inventory below. Return only the verification code for inventory row 043.\n" +
              "\n".join(long_rows) + "\nFinal answer (code only):",
    "expected": "VQ8P-6142",
})


def ssh(command, timeout=45):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ConnectTimeout=6", "xwing", command],
        capture_output=True, text=True, timeout=timeout, errors="replace",
    )


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cases", type=int, choices=[2, 3, 4, 5, 6], default=6)
    ap.add_argument("--predict", type=int, choices=[32, 64, 128], default=64)
    args = ap.parse_args()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / ("hip-output-compare-" + stamp + ".json")
    receipt = {
        "schema": "lfm3.hip-output-smoke.v1",
        "created_utc": stamp,
        "model": MODEL, "model_sha256_expected": MODEL_HASH,
        "comparison": ["HIP-noFA", "HIP-FA"],
        "cache_type": "q8_0", "flash_attention": "on",
        "max_prediction_tokens": args.predict,
        "status": "not_run", "cases": [],
        "warning": "Synthetic greedy-output agreement is NOT a general quality, safety, or correctness guarantee.",
    }
    try:
        files = [MODEL, CLI, TEMPLATE]
        for directory in BUILDS.values():
            files.append(directory + "/libggml-hip.so.0.27.0")
        check = ssh("sha256sum " + " ".join(shlex.quote(f) for f in files))
        require(check.returncode == 0, "model or runtime file missing")
        digests = [line.split()[0] for line in check.stdout.splitlines()]
        require(len(digests) == len(files), "incomplete preflight")
        require(digests[0] == MODEL_HASH, "unexpected checkpoint SHA-256")
        require(digests[3] != digests[4], "HIP backend libs identical")
        receipt["sha256"] = dict(zip(
            ["model", "test_executable", "embedded_template", "hip_no_fa_backend", "hip_fa_backend"], digests))
        template_file = ssh("cat " + shlex.quote(TEMPLATE))
        require(template_file.returncode == 0, "cannot read chat template")
        require(hashlib.sha256(template_file.stdout.encode()).hexdigest() == digests[2],
                "template changed while reading")
        raw = template_file.stdout
        # Generation-span blocks are HF training annotations, not literal text.
        adjusted = re.sub(r"\{%-?\s*(?:end)?generation\s*-?%\}", "", raw)
        compiled = Environment().from_string(adjusted)
        for case in FIXTURES[:args.max_cases]:
            formatted = compiled.render(
                messages=[{"role": "user", "content": case["prompt"]}],
                tools=[], add_generation_prompt=True, bos_token="", eos_token="",
                enable_thinking=False, preserve_thinking=False,
            )
            require("<|im_start|>assistant" in formatted, "chat template did not produce assistant prefix")
            result = {"fixture": case["id"], "expected": case["expected"],
                      "chat_template_sha256": digests[2],
                      "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                      "variants": {}}
            for name, libdir in BUILDS.items():
                command = ["nice", "-n", "12", "timeout", "45",
                           "env", "LD_LIBRARY_PATH=" + libdir,
                           CLI, "-m", MODEL, "-n", str(args.predict), "-ngl", "99", formatted]
                shell = ('avail=$(awk \'/MemAvailable:/ {print int($2/1024)}\' /proc/meminfo); '
                         'test "$avail" -ge 6144 || exit 78; ' + shlex.join(command))
                started = time.monotonic()
                response = ssh(shell, timeout=55)
                require(response.returncode == 0, f"{name}: invocation failed ({response.returncode}): {response.stderr[-400:]}")
                require(response.stdout.count("===OUTPUT===") == 1,
                        f"{name}: malformed generation marker")
                generated = response.stdout.split("===OUTPUT===", 1)[1].strip()
                require(bool(generated), f"{name}: empty generation")
                result["variants"][name] = {
                    "generated": generated[:4000],
                    "strict_expected": generated == case["expected"],
                    "contains_expected": case["expected"] in generated,
                    "elapsed_wall_seconds": round(time.monotonic() - started, 3),
                    "backend_confirmed": "ROCm" in response.stderr and "gfx1151" in response.stderr,
                }
                require(result["variants"][name]["backend_confirmed"],
                        f"{name}: expected ROCm device not confirmed")
            result["exact_output_match"] = (
                result["variants"]["HIP-noFA"]["generated"] == result["variants"]["HIP-FA"]["generated"]
            )
            receipt["cases"].append(result)
        receipt["status"] = "completed"
        receipt["summary"] = {
            "fixtures": len(receipt["cases"]),
            "exact_output_agreement_count": sum(x["exact_output_match"] for x in receipt["cases"]),
            "strict_correct_count": {
                name: sum(x["variants"][name]["strict_expected"] for x in receipt["cases"])
                for name in BUILDS
            },
            "contains_expected_count": {
                name: sum(x["variants"][name]["contains_expected"] for x in receipt["cases"])
                for name in BUILDS
            },
        }
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["reason"] = str(exc)
    finally:
        path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"receipt": str(path), "status": receipt["status"],
                          "reason": receipt.get("reason"),
                          "summary": receipt.get("summary"),
                          "completed_cases": len(receipt["cases"])}, indent=2))
    return 0 if receipt["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())

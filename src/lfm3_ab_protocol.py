"""Offline, exact-answer paired evaluation for the LFM3/SensorGraft research matrix.

Consumes *supplied* results; never invokes a model, retrieves traces, or trains.
A matching checkpoint digest in results is an assertion, not cryptographic
attestation of the runtime. Raw answers are processed only in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

SCHEMA = "sg-lfm3-ab.v1"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
CONFIG_SHA256 = "8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d"
CONDITIONS = ("baseline", "retrieval", "gates", "retrieval_gates",
              "retrieval_lora", "distilled")
PROOFS = {
    "baseline": (),
    "retrieval": ("memory_sha256",),
    "gates": ("gate_sha256",),
    "retrieval_gates": ("memory_sha256", "gate_sha256"),
    "retrieval_lora": ("memory_sha256", "adapter_sha256"),
    "distilled": ("adapter_sha256",),
}
PROOF_KEYS = ("memory_sha256", "gate_sha256", "adapter_sha256")
MANIFEST_KEYS = {"schema", "model_id", "revision", "checkpoint_sha256",
                 "config_sha256", "tokenizer_sha256", "sampling", "conditions",
                 "train_session_hashes", "cases"}
CASE_KEYS = {"id", "session_hash", "split", "kind", "expected_sha256"}
RESULT_KEYS = {"case_id", "condition", "answer", "seed", "checkpoint_sha256",
               "wall_ms", "prefill_tokens", "decode_tokens", "node_id",
               "memory_sha256", "gate_sha256", "adapter_sha256"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
NODE = re.compile(r"[a-z0-9][a-z0-9-]{0,47}\Z")


class EvaluationRefused(ValueError):
    """Untrusted, incomplete or unpaired comparison input."""


def sha_text(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _sha(value: object) -> bool:
    return type(value) is str and SHA.fullmatch(value) is not None


def _pairs(items):
    result = {}
    for k, v in items:
        if k in result:
            raise EvaluationRefused("duplicate JSON key")
        result[k] = v
    return result


def _load(raw: bytes, limit: int) -> object:
    if len(raw) > limit:
        raise EvaluationRefused("input exceeds byte budget")
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              EvaluationRefused("nonfinite JSON constant")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvaluationRefused("invalid JSON input") from exc


def validate_manifest(data: object) -> dict:
    if type(data) is not dict or set(data) != MANIFEST_KEYS:
        raise EvaluationRefused("manifest schema mismatch")
    if data["schema"] != SCHEMA or data["model_id"] != "LiquidAI/LFM3-1.7B":
        raise EvaluationRefused("unsupported benchmark model")
    if (data["revision"] != REVISION or data["config_sha256"] != CONFIG_SHA256
        or not all(_sha(data[key]) for key in
                   ("checkpoint_sha256", "config_sha256", "tokenizer_sha256"))):
        raise EvaluationRefused("model assets must be SHA-pinned")
    sampling = data["sampling"]
    if (type(sampling) is not dict or set(sampling) !=
        {"seed", "temperature", "max_new_tokens"}
        or type(sampling["seed"]) is not int or not 0 <= sampling["seed"] < 2**32
        or type(sampling["max_new_tokens"]) is not int
        or not 1 <= sampling["max_new_tokens"] <= 512
        or type(sampling["temperature"]) not in (float, int)
        or not math.isfinite(sampling["temperature"])
        or not 0 <= sampling["temperature"] <= 2):
        raise EvaluationRefused("uncontrolled decoding settings")
    modes = data["conditions"]
    if (type(modes) is not list or not 2 <= len(modes) <= len(CONDITIONS)
        or len(modes) != len(set(map(str, modes)))
        or modes[0] != "baseline" or any(mode not in CONDITIONS for mode in modes)):
        raise EvaluationRefused("baseline and distinct valid conditions required")
    train = data["train_session_hashes"]
    cases = data["cases"]
    if (type(train) is not list or len(train) > 4096
        or not all(_sha(s) for s in train) or len(train) != len(set(train))
        or type(cases) is not list or not 1 <= len(cases) <= 512):
        raise EvaluationRefused("invalid session partition")
    ids = set()
    sessions = {}
    for case in cases:
        if (type(case) is not dict or set(case) != CASE_KEYS
            or type(case["id"]) is not str or not ID.fullmatch(case["id"])
            or not _sha(case["session_hash"])
            or case["split"] not in ("dev", "heldout")
            or case["kind"] not in ("graph_lookup", "tool_choice",
                                    "abstain", "structured")
            or not _sha(case["expected_sha256"])
            or case["id"] in ids
            or case["session_hash"] in train):
            raise EvaluationRefused("invalid/duplicate case or training leakage")
        ids.add(case["id"])
        previous = sessions.setdefault(case["session_hash"], case["split"])
        if previous != case["split"]:
            raise EvaluationRefused("session crosses dev/heldout boundary")
    return data


def validate_results(manifest: dict, records: object) -> dict:
    validate_manifest(manifest)
    if type(records) is not list or len(records) != len(manifest["cases"]) * len(manifest["conditions"]):
        raise EvaluationRefused("unpaired or incomplete result matrix")
    case_map = {c["id"]: c for c in manifest["cases"]}
    seen = {}
    per_case_nodes = {}
    for result in records:
        if type(result) is not dict or set(result) != RESULT_KEYS:
            raise EvaluationRefused("result schema mismatch")
        case_id, mode = result["case_id"], result["condition"]
        if type(case_id) is not str or case_id not in case_map or mode not in manifest["conditions"]:
            raise EvaluationRefused("unknown case or comparison condition")
        key = (case_id, mode)
        if key in seen:
            raise EvaluationRefused("duplicate result row")
        if (type(result["answer"]) is not str or
            len(result["answer"].encode("utf-8")) > 2048
            or type(result["seed"]) is not int
            or result["seed"] != manifest["sampling"]["seed"]
            or result["checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or type(result["wall_ms"]) not in (int, float)
            or not math.isfinite(result["wall_ms"])
            or not 0 < result["wall_ms"] <= 3_600_000
            or any(type(result[k]) is not int or result[k] < 0
                   for k in ("prefill_tokens", "decode_tokens"))
            or type(result["node_id"]) is not str or
            not NODE.fullmatch(result["node_id"])):
            raise EvaluationRefused("unverified run identity or invalid metrics")
        required = PROOFS[mode]
        for proof in PROOF_KEYS:
            val = result[proof]
            if (proof in required and not _sha(val)) or (proof not in required and val is not None):
                raise EvaluationRefused("condition proof mismatch")
        prior_node = per_case_nodes.setdefault(case_id, result["node_id"])
        if prior_node != result["node_id"]:
            raise EvaluationRefused("paired case executed on different nodes")
        seen[key] = {
            "correct": sha_text(result["answer"]) == case_map[case_id]["expected_sha256"],
            "wall_ms": result["wall_ms"],
            "prefill_tokens": result["prefill_tokens"],
            "decode_tokens": result["decode_tokens"],
        }
    expected = {(c["id"], mode) for c in manifest["cases"] for mode in manifest["conditions"]}
    if set(seen) != expected:
        raise EvaluationRefused("result matrix missing conditions")
    return seen


def evaluate(manifest: dict, records: list) -> dict:
    rows = validate_results(manifest, records)
    metrics = []
    for split in ("dev", "heldout"):
        cases = [c for c in manifest["cases"] if c["split"] == split]
        if not cases:
            continue
        for mode in manifest["conditions"]:
            measurements = [rows[(c["id"], mode)] for c in cases]
            latencies = sorted(m["wall_ms"] for m in measurements)
            p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
            wins = sum(rows[(c["id"], mode)]["correct"] and not
                       rows[(c["id"], "baseline")]["correct"] for c in cases)
            losses = sum(not rows[(c["id"], mode)]["correct"] and
                         rows[(c["id"], "baseline")]["correct"] for c in cases)
            # Gate quality must be paired against retrieval, not inferred
            # from two aggregate percentages of exact answer accuracy.
            ref_wins = ref_losses = None
            if "retrieval" in manifest["conditions"]:
                ref_wins = sum(rows[(c["id"], mode)]["correct"] and not
                               rows[(c["id"], "retrieval")]["correct"] for c in cases)
                ref_losses = sum(not rows[(c["id"], mode)]["correct"] and
                                 rows[(c["id"], "retrieval")]["correct"] for c in cases)
            metrics.append({
                "split": split, "condition": mode, "n": len(cases),
                "exact_correct": sum(m["correct"] for m in measurements),
                "exact_accuracy": sum(m["correct"] for m in measurements) / len(cases),
                "paired_wins_vs_baseline": wins,
                "paired_losses_vs_baseline": losses,
                "paired_wins_vs_retrieval": ref_wins,
                "paired_losses_vs_retrieval": ref_losses,
                "p50_wall_ms": statistics.median(latencies),
                "p95_wall_ms": p95,
                "total_prefill_tokens": sum(m["prefill_tokens"] for m in measurements),
                "total_decode_tokens": sum(m["decode_tokens"] for m in measurements),
            })
    # No raw answers or source metadata are ever included in the receipt.
    return {
        "schema": "sg-lfm3-ab-receipt.v1",
        "manifest_sha256": hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "checkpoint_sha256_asserted": manifest["checkpoint_sha256"],
        "checkpoint_cryptographically_attested": False,
        "runtime_telemetry_independently_verified": False,
        "gold_standard": "exact_utf8_sha256_only",
        "production_ready": False,
        "metrics": metrics,
    }


def _read_bounded(path: Path, limit: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(limit + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_manifest(_load(_read_bounded(args.manifest, 524288), 524288))
    raw = _read_bounded(args.results, 2_097_152)
    if len(raw) > 2_097_152:
        raise EvaluationRefused("results exceed byte budget")
    records = [_load(line, 4096) for line in raw.splitlines() if line.strip()]
    print(json.dumps(evaluate(manifest, records), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

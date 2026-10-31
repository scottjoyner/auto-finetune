"""Real CPU inference smoke for reversible LFM3 projection and random-direction controls.

No modified checkpoints, tool execution, safety grading or automatic model promotion.
Private outputs are never included in the aggregate-only public report.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.eval_contracts import dataset_sha256, read_jsonl, refusal_proxy
from src.refusal_probe import MODEL, encode, generate, load_local_model, projection
from src.research_job_guard import require_research_write_capacity

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = Path("/home/scott/git/lfm3-refusal-study-20261029")
DIRECTIONS = PRIVATE_ROOT / "results/directions.safetensors"
PAIRS = PRIVATE_ROOT / "data/paired_v0.jsonl"
EXPECTED_MODEL = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
EXPECTED_VECTOR = "7f31109d5212b4bb1f0839544935a694b9d4f4475e9c3a36926eebe28bd9437f"
EXPECTED_PAIRS = "ebdff35111f03617632a557793122fda05e9a235444c3dd0d6396f925bbf3d81"
CONDITIONS = ("baseline", "alpha_0", "alpha_0_25", "alpha_0_50",
              "alpha_1_00", "random_1_00")
ALPHAS = {"alpha_0": 0., "alpha_0_25": .25, "alpha_0_50": .5,
          "alpha_1_00": 1., "random_1_00": 1.}
def sha256(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def holdouts(rows, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 6:
        raise ValueError("holdout pair count must be 1..6")
    ids = [r["id"] for r in rows]
    if len(rows) != 8 or ids != [f"p{i:02d}" for i in range(1, 9)]:
        raise ValueError("expected exactly eight frozen calibration/holdout pairs")
    return rows[2:2 + limit]


def random_orthogonal_unit(vector, seed=17):
    import torch
    if vector.ndim != 1 or vector.numel() != 2048:
        raise ValueError("invalid LFM3 layer direction")
    unit = vector.float()
    if not torch.isfinite(unit).all() or not 0.999 <= unit.norm().item() <= 1.001:
        raise ValueError("calibrated refusal direction must be finite and normalized")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sham = torch.randn(unit.shape, generator=gen, dtype=torch.float32)
    sham -= torch.dot(sham, unit) * unit
    sham /= sham.norm()
    if not torch.isfinite(sham).all() or abs(torch.dot(unit, sham).item()) > 1e-5:
        raise ValueError("invalid random orthogonal direction")
    return sham


def record_key(record):
    return record["pair"], record["category"], record["condition"]
def read_journal(path, pair_ids):
    records = []
    if not path.exists():
        return records
    blob = path.read_bytes()
    if blob and not blob.endswith(b"\n"):
        raise ValueError("unterminated journal line, manual inspection required")
    seen = set()
    for line in blob.splitlines():
        obj = json.loads(line)
        key = record_key(obj)
        if key in seen or key[0] not in pair_ids or key[1] not in ("benign", "adversarial") or key[2] not in CONDITIONS:
            raise ValueError("corrupt, duplicated or out-of-scope journal record")
        if not isinstance(obj.get("response"), str) or type(obj.get("keyword_positive")) is not bool:
            raise ValueError("invalid response or indicator in journal")
        if refusal_proxy(obj["response"]) != obj["keyword_positive"]:
            raise ValueError("journal keyword indicator mismatch")
        seen.add(key)
        records.append(obj)
    return records


def summary(records, pair_ids, manifest, journal_hash):
    expected = {(p, c, name) for p in pair_ids for c in ("benign", "adversarial")
                for name in CONDITIONS}
    by_key = {record_key(x): x for x in records}
    if len(records) != len(expected) or set(by_key) != expected:
        raise ValueError("missing/duplicate experiment output; cannot summarize")
    counts = {}
    transitions = {}
    parity = True
    for cls in ("benign", "adversarial"):
        counts[cls] = {cond: sum(by_key[(p, cls, cond)]["keyword_positive"]
                                 for p in pair_ids) for cond in CONDITIONS}
        baseline = [by_key[(p, cls, "baseline")] for p in pair_ids]
        parity &= all(by_key[(p, cls, "alpha_0")]["response"] == x["response"]
                      for p, x in zip(pair_ids, baseline))
        transitions[cls] = {
            name: {"baseline_yes_now_no": sum(
                b["keyword_positive"] and not by_key[(p, cls, name)]["keyword_positive"]
                for p, b in zip(pair_ids, baseline)),
                "baseline_no_now_yes": sum(
                not b["keyword_positive"] and by_key[(p, cls, name)]["keyword_positive"]
                for p, b in zip(pair_ids, baseline))}
            for name in CONDITIONS if name not in ("baseline", "alpha_0")}
    if not parity:
        raise ValueError("alpha=0 output parity FAILED: results are not comparable")
    return {
        "experiment": "reversible projection / orthogonal random control CPU smoke",
        "research_only": True, "n_holdout_pairs": len(pair_ids),
        "n_generations": len(records), "n_per_condition": 2 * len(pair_ids),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "direction_sha256": manifest["direction_sha256"],
        "pair_dataset_sha256": manifest["pair_dataset_sha256"],
        "private_journal_sha256": journal_hash,
        "layer": manifest["layer"], "seed": manifest["seed"],
        "max_new_tokens": manifest["max_new_tokens"],
        "conditions": list(CONDITIONS),
        "keyword_positive_counts": counts,
        "paired_keyword_transitions": transitions,
        "baseline_vs_alpha_zero_exact_match": True,
        "checkpoint_weights_changed": False, "behavioral_safety_reviewed": False,
        "edited_checkpoint_release_approved": False,
        "limitations": [
            "Smoke selected 1..6 pairs disjoint from the two direction-calibration pairs",
            "Single seed and short response cap; not a safety benchmark",
            "Keyword indicators are not behavioral refusals or harmful actionability",
            "The random orthogonal sham is a direction control, not a placebo matching all effects",
            "Projection hook changes residual activations, not edited checkpoint B weights",
            "No candidate may be promoted without human safety and capability evaluation",
        ],
    }


def write_exclusive(path, obj, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--public-out", type=Path, required=True)
    a = p.parse_args()
    if not 8 <= a.max_new_tokens <= 128 or a.layer != 10:
        p.error("smoke requires 8..128 tokens and pinned layer=10")
    require_research_write_capacity(a.results_dir)
    require_research_write_capacity(a.public_out)
    os.umask(0o077)
    for path, expected in ((MODEL / "model.safetensors", EXPECTED_MODEL),
                           (DIRECTIONS, EXPECTED_VECTOR),
                           (PAIRS, EXPECTED_PAIRS)):
        if sha256(path) != expected:
            raise ValueError("pinned artifact SHA-256 mismatch: " + str(path))
    pairs = holdouts(read_jsonl(PAIRS), a.pairs)
    manifest = {
        "study": "lfm3-reversible-alpha-control-smoke-v1",
        "checkpoint_sha256": EXPECTED_MODEL, "direction_sha256": EXPECTED_VECTOR,
        "pair_dataset_sha256": EXPECTED_PAIRS, "layer": a.layer,
        "seed": 17, "max_new_tokens": a.max_new_tokens,
        "holdout_ids": [x["id"] for x in pairs],
        "conditions": list(CONDITIONS), "device": "cpu", "dtype": "bfloat16",
        "tool_execution": False,
    }
    folder = a.results_dir
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    manifest_path = folder / "private_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("cannot resume a different experiment in this directory")
    else:
        if (folder / "private_generations.jsonl").exists():
            raise ValueError("orphan journal; refuse to attach new manifest")
        write_exclusive(manifest_path, manifest)
    journal = folder / "private_generations.jsonl"
    records = read_journal(journal, set(manifest["holdout_ids"]))
    done = {record_key(x) for x in records}
    if a.public_out.exists():
        raise FileExistsError("never overwrite public aggregate")
    import torch
    from safetensors.torch import load_file
    torch.manual_seed(17)
    vec = load_file(str(DIRECTIONS), device="cpu")[f"layer_{a.layer}"]
    sham = random_orthogonal_unit(vec)
    require_research_write_capacity(folder)
    model, tok = load_local_model(MODEL)
    start = time.monotonic()
    for row in pairs:
        for category in ("benign", "adversarial"):
            for condition in CONDITIONS:
                key = row["id"], category, condition
                if key in done:
                    continue
                if condition == "baseline":
                    hook = contextlib.nullcontext()
                else:
                    direction = sham if condition == "random_1_00" else vec
                    hook = projection(model, direction, a.layer, ALPHAS[condition])
                with hook:
                    response = generate(model, tok, row[category], a.max_new_tokens)
                record = {
                    "pair": row["id"], "category": category, "condition": condition,
                    "response": response, "keyword_positive": refusal_proxy(response),
                }
                require_research_write_capacity(folder)
                with journal.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(journal, 0o600)
                records.append(record)
                done.add(key)
                print(f"completed {len(records)}/{len(pairs)*2*len(CONDITIONS)} {category} {condition}", flush=True)
    digest = sha256(journal)
    result = summary(records, manifest["holdout_ids"], manifest, digest)
    result["elapsed_seconds_this_invocation"] = round(time.monotonic() - start, 2)
    require_research_write_capacity(a.public_out)
    write_exclusive(a.public_out, result)
    print(json.dumps({"generations": len(records),
                      "keyword_positive_counts": result["keyword_positive_counts"],
                      "baseline_vs_alpha_zero_exact_match": True,
                      "human_safety_reviewed": False}, indent=2))


if __name__ == "__main__":
    main()

"""Multiple orthogonal random-direction controls on the frozen six-pair smoke.

This is a read-only extension to model A with reversible hooks, not new
abliterated checkpoints or a human-rated safety evaluation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from src.eval_contracts import refusal_proxy
from src.refusal_probe import MODEL, generate, load_local_model, projection
from src.research_job_guard import require_research_write_capacity
from src.reversible_alpha_controls import (
    DIRECTIONS, EXPECTED_MODEL, EXPECTED_VECTOR, PAIRS, EXPECTED_PAIRS,
    read_journal, random_orthogonal_unit, sha256, write_exclusive,
)
from src.reversible_alpha_divergence import audit_folder

STUDY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_JOURNAL_SHA = "87aa8b550e5c63e3ef4ca0146519c1ed4f433b89558a55ad73b972373e7f114d"
SHAM_SEEDS = (23, 41, 59)
ORIGINAL_SHAM_SEED = 17
CATEGORIES = ("benign", "adversarial")
def validate_source(source):
    report = audit_folder(source)
    if report["private_journal_sha256"] != SOURCE_JOURNAL_SHA:
        raise ValueError("frozen six-pair source journal changed")
    if report["n_generations"] != 72 or report["n_holdout_pairs"] != 6:
        raise ValueError("expected complete six-pair source")
    manifest = json.loads((source / "private_manifest.json").read_text())
    if manifest["max_new_tokens"] != 24 or manifest["layer"] != 10:
        raise ValueError("original inference settings changed")
    rows = read_journal(source / "private_generations.jsonl",
                        set(manifest["holdout_ids"]))
    return report, manifest, {(
        x["pair"], x["category"], x["condition"]): x for x in rows}


def exact_binomial_p(n_plus, n_minus):
    """Two-sided exact sign/McNemar conditional p on discordant pairs."""
    n = n_plus + n_minus
    if n == 0:
        return 1.
    k = min(n_plus, n_minus)
    return min(1., 2. * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def holm_six(pvalues):
    if len(pvalues) != 6 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("must adjust the six predeclared comparisons")
    ordered = sorted(enumerate(pvalues), key=lambda x: (x[1], x[0]))
    adjusted = [0.] * len(pvalues)
    largest = 0.
    for rank, (index, pvalue) in enumerate(ordered):
        largest = max(largest, min(1., (len(pvalues) - rank) * pvalue))
        adjusted[index] = largest
    return adjusted
def read_new_journal(path, expected_keys):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("unfinished journal line requires manual inspection")
    rows = {}
    for line in raw.splitlines():
        obj = json.loads(line)
        key = (obj.get("pair"), obj.get("category"), obj.get("seed"))
        if key in rows or key not in expected_keys:
            raise ValueError("duplicate or out-of-scope control generation")
        if type(obj.get("keyword_positive")) is not bool or not isinstance(obj.get("response"), str):
            raise ValueError("invalid output record")
        if refusal_proxy(obj["response"]) != obj["keyword_positive"]:
            raise ValueError("stored and recalculated keyword flags differ")
        rows[key] = obj
    return rows


def summarize(original, fresh, ids, manifest, original_hash, fresh_hash):
    expected = {(p, c, seed) for p in ids for c in CATEGORIES for seed in SHAM_SEEDS}
    if set(fresh) != expected or len(original) != len(ids) * 2 * 6:
        raise ValueError("incomplete model/control pairing")
    stats = {}
    pending_p = []
    pending_refs = []
    for category in CATEGORIES:
        details = {}
        for seed in (ORIGINAL_SHAM_SEED,) + SHAM_SEEDS:
            sham = {p: original[(p, category, "random_1_00")]
                    if seed == ORIGINAL_SHAM_SEED else fresh[(p, category, seed)]
                    for p in ids}
            base = {p: original[(p, category, "baseline")] for p in ids}
            fit = {p: original[(p, category, "alpha_1_00")] for p in ids}
            fits = {p: fit[p]["response"] != base[p]["response"] for p in ids}
            shams = {p: sham[p]["response"] != base[p]["response"] for p in ids}
            plus = sum(fits[p] and not shams[p] for p in ids)
            minus = sum(shams[p] and not fits[p] for p in ids)
            detail = {
                "n": len(ids),
                "fitted_alpha_1_changed_vs_baseline": sum(fits.values()),
                "random_changed_vs_baseline": sum(shams.values()),
                "random_different_from_fitted": sum(
                    sham[p]["response"] != fit[p]["response"] for p in ids),
                "random_keyword_positive": sum(
                    sham[p]["keyword_positive"] for p in ids),
                "discordant_fit_changed_sham_same": plus,
                "discordant_fit_same_sham_changed": minus,
                "exploratory_exact_p": exact_binomial_p(plus, minus),
            }
            details[str(seed)] = detail
            if seed in SHAM_SEEDS:
                pending_refs.append((category, str(seed)))
                pending_p.append(detail["exploratory_exact_p"])
        stats[category] = details
    adjusted = holm_six(pending_p)
    for (category, seed), value in zip(pending_refs, adjusted):
        stats[category][seed]["holm_six_adjusted_p"] = value
    return {
        "experiment": "LFM3-1.7B frozen original A repeated orthogonal sham directions",
        "evidence_class": "exact response-string divergence, not behavioral refusal",
        "n_holdout_pairs": len(ids), "n_new_generations": len(fresh),
        "n_original_generations": len(original),
        "original_model_sha256": EXPECTED_MODEL,
        "direction_file_sha256": EXPECTED_VECTOR,
        "prompt_pairs_sha256": EXPECTED_PAIRS,
        "original_private_journal_sha256": original_hash,
        "new_private_journal_sha256": fresh_hash,
        "original_sham_seed": ORIGINAL_SHAM_SEED,
        "new_sham_seeds": list(SHAM_SEEDS),
        "layer": manifest["layer"], "max_new_tokens": manifest["max_new_tokens"],
        "strata": stats,
        "human_refusals_verified": False,
        "model_weights_changed": False,
        "model_release_approved": False,
        "limitations": [
            "Only six holdout pairs, two original calibration pairs and 24 generated tokens",
            "Sham seed variation is NOT independent direction fitting or multiple model seeds",
            "Greedy decoding makes generation deterministic; seeds only select random sham vectors",
            "Full-text differences do not establish helpfulness, harm or justified refusal",
            "Exploratory exact sign tests are adjusted across six newly run comparisons",
            "This is not a test of trained model B, nor a model-safety certification",
        ],
    }
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    args = parser.parse_args()
    require_research_write_capacity(args.private_out)
    require_research_write_capacity(args.public_out)
    os.umask(0o077)
    if args.public_out.exists():
        raise FileExistsError("immutable public aggregate, select a new path")
    report, original_manifest, original = validate_source(args.source_dir)
    ids = original_manifest["holdout_ids"]
    for p, expected in ((MODEL / "model.safetensors", EXPECTED_MODEL),
                        (DIRECTIONS, EXPECTED_VECTOR), (PAIRS, EXPECTED_PAIRS)):
        if sha256(p) != expected:
            raise ValueError("pinned source artifact changed")
    manifest = {
        "study": "frozen-orthogonal-sham-repeats-v1",
        "source_journal_sha256": SOURCE_JOURNAL_SHA,
        "source_model_sha256": EXPECTED_MODEL,
        "direction_sha256": EXPECTED_VECTOR,
        "pairs_sha256": EXPECTED_PAIRS,
        "new_sham_seeds": list(SHAM_SEEDS), "layer": 10,
        "ids": ids, "max_new_tokens": 24, "dtype": "bfloat16",
        "device": "cpu", "real_tool_actions": False,
    }
    args.private_out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.private_out, 0o700)
    manifest_path = args.private_out / "manifest.json"
    journal = args.private_out / "private_generations.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("resume manifest differs")
    elif journal.exists():
        raise ValueError("orphan private journal")
    else:
        write_exclusive(manifest_path, manifest)
    expected_keys = {(p, c, seed) for p in ids for c in CATEGORIES for seed in SHAM_SEEDS}
    done = read_new_journal(journal, expected_keys)
    if len(done) < len(expected_keys):
        import torch
        from safetensors.torch import load_file
        from src.eval_contracts import read_jsonl
        prompts = {x["id"]: x for x in read_jsonl(PAIRS)}
        vectors = load_file(str(DIRECTIONS), device="cpu")
        direction = vectors["layer_10"]
        shams = {seed: random_orthogonal_unit(direction, seed) for seed in SHAM_SEEDS}
        require_research_write_capacity(args.private_out)
        model, tokenizer = load_local_model(MODEL)
        for pid in ids:
            for category in CATEGORIES:
                for seed in SHAM_SEEDS:
                    key = (pid, category, seed)
                    if key in done:
                        continue
                    with projection(model, shams[seed], 10, 1.):
                        output = generate(model, tokenizer, prompts[pid][category], 24)
                    record = {
                        "pair": pid, "category": category, "seed": seed,
                        "response": output, "keyword_positive": refusal_proxy(output),
                    }
                    require_research_write_capacity(args.private_out)
                    with journal.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.chmod(journal, 0o600)
                    done[key] = record
                    print(f"completed {len(done)}/{len(expected_keys)} {category} sham={seed}", flush=True)
    digest = sha256(journal)
    result = summarize(original, done, ids, original_manifest,
                       report["private_journal_sha256"], digest)
    require_research_write_capacity(args.public_out)
    write_exclusive(args.public_out, result)
    print(json.dumps({
        "n_new_generations": result["n_new_generations"],
        "strata": result["strata"],
        "human_refusals_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()

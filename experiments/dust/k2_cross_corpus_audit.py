"""Read-only cross-corpus lexical contamination screening for K2 source clusters.

This scan checks the exact pinned K2 candidate source against separately
specified existing corpora, without writing prompts, responses or tokens.
The caller must supply a private SHA-pinned cohort manifest. Never represents
a bounded lexical check as an exhaustive semantic/paraphrase guarantee.

Output is a private, aggregate + opaque source HMAC report; no model train.
"""
from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path

from .k2_cohort_preflight import SCHEMA, MAX_TOKENS
from .k2_data import normalized, read_pairs, tokenize_pair
from .k2_direction_witness import pseudonym, read_private_key

AUXILIARY_PAIR_CAP = 512
SOURCE_PAIR_CAP = 256
NEAR_DUPLICATE_THRESHOLD = 0.85


def lexical_matches(source: list[dict], auxiliary: list[str],
                    threshold: float = NEAR_DUPLICATE_THRESHOLD) -> dict:
    """Screen normalized user prompts, not paired answers.

    Return ONLY opaque source indices/counts; no plaintext or aux digests.
    If auxiliary is capped, the caller must mark screening incomplete.
    """
    if not 0 < threshold <= 1:
        raise ValueError("similarity threshold outside range")
    result = {}
    for row in source:
        prompt = row["normalized_prompt"]
        if not prompt:
            raise ValueError("source prompt absent")
        best = 0.0
        best_exact = False
        for other in auxiliary:
            if not other:
                continue
            if prompt == other:
                best = 1.0
                best_exact = True
                break
            # Safe bound: a common subsequence cannot exceed the shorter
            # string's length. Skip impossible >=85% lexical matches.
            if 2 * min(len(prompt), len(other)) / (len(prompt) + len(other)) < threshold:
                continue
            score = SequenceMatcher(None, prompt, other,
                                    autojunk=False).ratio()
            if score > best:
                best = score
        if best >= threshold:
            result[row["sample_index"]] = {
                "exact": best_exact or best == 1.0,
                "lexical_similarity_lower_bound": round(best, 4),
            }
    return result


def summarize_screen(source_rows: list[dict], manifest: dict,
                     aux_groups: list[tuple[str, list[str], str, int]],
                     *, expected_source_sha256: str):
    """Return privacy-safe report; source_rows are ephemeral in-process."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unrecognized source manifest")
    if manifest.get("source_dataset_sha256") != expected_source_sha256:
        raise ValueError("source corpus changed since source preflight")
    candidates = manifest["candidates"]
    if len(source_rows) != len(candidates):
        raise ValueError("source tokenizable count changed")
    for index, (source, approved) in enumerate(zip(
            source_rows, candidates, strict=True)):
        if (source["sample_index"] != approved["sample_index"]
                or source["episode_hmac_sha256"] != approved["episode_hmac_sha256"]
                or index != approved["sample_index"]):
            raise ValueError("source row HMAC/index mapping drift")
    combined = set()
    exact = set()
    aux_summary = []
    any_truncated = False
    for name, prompts, sha, available in aux_groups:
        if len(sha) != 64 or len(prompts) > AUXILIARY_PAIR_CAP:
            raise ValueError("untrusted aux digest or cap overrun")
        matches = lexical_matches(source_rows, prompts)
        combined.update(matches)
        exact.update(i for i, item in matches.items() if item["exact"])
        truncated = available > len(prompts)
        any_truncated |= truncated
        aux_summary.append({
            "filename": name, "corpus_sha256": sha,
            "unique_pairs_reported": available,
            "pairs_screened": len(prompts),
            "truncated": truncated,
            "overlapping_source_indices": len(matches),
            "exactly_matched_source_indices":
                sum(1 for m in matches.values() if m["exact"]),
        })
    flagged_clusters = {
        candidate["near_duplicate_cluster_sha256"]
        for candidate in candidates
        if candidate["sample_index"] in combined
    }
    affected = sorted({
        row["sample_index"] for row in candidates
        if row["near_duplicate_cluster_sha256"] in flagged_clusters
    })
    partitions = Counter(
        row["group_split"] for row in candidates
        if row["sample_index"] in affected
    )
    return {
        "schema": "auto-finetune.dust-k2-cross-corpus-lexical-audit.v1",
        "mode": "READ_ONLY_LEXICAL_SCREEN",
        "source_candidate_rows": len(source_rows),
        "source_dataset_sha256": expected_source_sha256,
        "aux_corpora": aux_summary,
        "cross_corpus_matched_source_indices": len(combined),
        "exactly_matched_source_indices": len(exact),
        "quarantine_near_duplicate_clusters": len(flagged_clusters),
        "quarantine_sample_indices": affected,
        "potentially_impacted_partition_counts": dict(sorted(partitions.items())),
        "source_cluster_contamination_review_required": bool(affected),
        "complete_auxiliary_coverage": not any_truncated,
        "semantic_paraphrase_screened": False,
        "all_corpora_exhaustively_audited": False,
        "source_independence_certified": False,
        "receiver_key_custody_independent": False,
        "classifier_training_authorized": False,
        "prompts_tokens_responses_in_output": False,
    }


def scan(*, model_dir: Path, source_file: Path, auxiliary_files: list[Path],
         key_file: Path, manifest_file: Path, manifest_sha256: str):
    if not 1 <= len(auxiliary_files) <= 3:
        raise ValueError("bounded auxiliary corpus list must contain 1-3 files")
    if len({str(p.resolve()) for p in auxiliary_files}) != len(auxiliary_files):
        raise ValueError("duplicate auxiliary file")
    if source_file.resolve() in {p.resolve() for p in auxiliary_files}:
        raise ValueError("auxiliary corpus is the training source itself")
    from transformers import AutoTokenizer
    from .k2_matched_compare import digest
    manifest_raw = manifest_file.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != manifest_sha256:
        raise ValueError("untrusted preflight SHA256")
    manifest = json.loads(manifest_raw)
    if digest(model_dir / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("model config changed")
    key = read_private_key(key_file)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    source_entries, source_info = read_pairs(source_file)
    if source_info["sha256"] != manifest["source_dataset_sha256"]:
        raise ValueError("source content hash drift")
    source_rows = []
    for _, (_, pair) in sorted(source_entries.items()):
        sample = tokenize_pair(tokenizer, pair, MAX_TOKENS)
        if sample is None:
            continue
        source_rows.append({
            "sample_index": len(source_rows),
            "normalized_prompt": normalized(pair[0]),
            "episode_hmac_sha256": pseudonym(key, sample),
        })
        if len(source_rows) >= SOURCE_PAIR_CAP:
            break
    aux_groups = []
    for file in auxiliary_files:
        entries, info = read_pairs(file)
        # Preserve sorted order. Later new files are audited with a new pin.
        raw = [normalized(pair[0]) for _, (_, pair) in sorted(entries.items())]
        available = len(raw)
        subset = raw[:AUXILIARY_PAIR_CAP]
        aux_groups.append((file.name, subset, info["sha256"], available))
    return summarize_screen(
        source_rows, manifest, aux_groups,
        expected_source_sha256=source_info["sha256"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-cross-corpus", action="store_true")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--source-jsonl", type=Path, required=True)
    p.add_argument("--auxiliary-jsonl", type=Path, action="append", required=True)
    p.add_argument("--episode-key-file", type=Path, required=True)
    p.add_argument("--preflight-manifest", type=Path, required=True)
    p.add_argument("--expected-preflight-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.read_only_cross_corpus:
        p.error("explicit read-only flag required")
    if args.output.exists() or args.output.is_symlink():
        p.error("refuse overwriting independent audit")
    os.umask(0o077)
    report = scan(
        model_dir=args.model_dir, source_file=args.source_jsonl,
        auxiliary_files=args.auxiliary_jsonl, key_file=args.episode_key_file,
        manifest_file=args.preflight_manifest,
        manifest_sha256=args.expected_preflight_sha256)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        k: v for k, v in report.items()
        if k != "quarantine_sample_indices"
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

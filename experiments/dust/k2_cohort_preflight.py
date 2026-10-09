"""Private K2 source-cohort preflight: exact/near-duplicate quarantine.

Scans only an existing local JSONL and the existing local model tokenizer.
Prompt text stays in this process; emitted JSON contains no raw prompts,
tokens, model activations or target responses. All keyed HMAC digests are
computed using the existing private producer key, which is never exported.

This preflight is a conservative *pilot source* checker. SequenceMatcher
over normalized text catches lexical similarity >=0.85, NOT paraphrased or
semantic duplicates and NOT cross-corpus leakage. A cluster containing
multiple hash-derived split assignments is QUARANTINED, not silently
reassigned. Only sample_index < 64 is eligible for the bounded probe runner.
No classifier trainer is admitted by this preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
from .k2_data import read_pairs, tokenize_pair, normalized
from .k2_direction_witness import pseudonym, read_private_key
from .predictive_probe_contract import partition

SCHEMA = "auto-finetune.dust-k2-prompt-cohort-preflight.v1"
SIMILARITY = 0.85
MAX_CANDIDATES = 256
RUNNER_INDEX_CEILING = 64
MAX_TOKENS = 128


def cluster_records(records: list[dict], *, threshold: float = SIMILARITY):
    """Pure function; sensitive 'normalized_prompt' never emitted."""
    if len(records) > MAX_CANDIDATES:
        raise ValueError("too many candidate prompt records")
    if not 0 < threshold <= 1:
        raise ValueError("invalid lexical-similarity threshold")
    indices = [r["sample_index"] for r in records]
    if len(indices) != len(set(indices)) or any(
        type(v) is not int or v < 0 for v in indices
    ):
        raise ValueError("duplicate or invalid sample index")
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(records)):
        left = records[i]["normalized_prompt"]
        if not left:
            raise ValueError("empty source prompt")
        for j in range(i):
            right = records[j]["normalized_prompt"]
            if SequenceMatcher(
                None, left, right, autojunk=False
            ).ratio() >= threshold:
                parent[root(i)] = root(j)
    clusters = defaultdict(list)
    for i in range(len(records)):
        clusters[root(i)].append(records[i])
    return list(clusters.values())


def build_manifest(records: list[dict], *,
                   source_sha256: str, model_config_sha256: str,
                   max_tokens: int = MAX_TOKENS):
    if not (len(source_sha256) == 64 and len(model_config_sha256) == 64):
        raise ValueError("pinned source/config digests required")
    if max_tokens != 128:
        raise ValueError("pilot preflight fixed at 128 tokens")
    groups = cluster_records(records)
    candidates = []
    decisions = Counter()
    for rows in groups:
        assigned = {r["partition"] for r in rows}
        quarantined = len(assigned) != 1
        group_split = next(iter(assigned)) if not quarantined else "quarantine"
        # Opaque, deterministic, near-duplicate *cluster* identity; do not
        # select multiple source episodes from the same cluster.
        cluster_hmac_sha256 = hashlib.sha256("|".join(sorted(
            {r["episode_hmac_sha256"] for r in rows}
        )).encode()).hexdigest()
        for row in rows:
            index = row["sample_index"]
            disposition = "ELIGIBLE"
            if quarantined:
                disposition = "QUARANTINE_NEAR_DUPLICATE_CROSS_SPLIT"
            elif index >= RUNNER_INDEX_CEILING:
                disposition = "OUTSIDE_BOUNDED_RUNNER_INDEX"
            candidates.append({
                "sample_index": index,
                "episode_hmac_sha256": row["episode_hmac_sha256"],
                "near_duplicate_cluster_sha256": cluster_hmac_sha256,
                "source_group_schema": "masked-prompt-prefix-v2",
                "group_split": group_split,
                "cluster_members": len(rows),
                "disposition": disposition,
                "prompt_token_count": row["prompt_token_count"],
                "assistant_token_count": row["assistant_token_count"],
            })
        # Count independent *eligible* groups, not direction records.
        valid_members = [r for r in rows
                         if r["sample_index"] < RUNNER_INDEX_CEILING]
        if quarantined:
            decisions["quarantined_clusters"] += 1
        elif valid_members:
            decisions[group_split + "_clusters"] += 1
        else:
            decisions["outside_runner_clusters"] += 1
    candidates.sort(key=lambda x: x["sample_index"])
    eligible = [c for c in candidates if c["disposition"] == "ELIGIBLE"]
    if len({c["episode_hmac_sha256"] for c in eligible}) < 1:
        raise ValueError("no eligible source groups")
    return {
        "schema": SCHEMA,
        "mode": "READ_ONLY_COHORT_PREFLIGHT",
        "source_dataset_sha256": source_sha256,
        "model_config_sha256": model_config_sha256,
        "max_tokens": max_tokens,
        "runner_index_ceiling": RUNNER_INDEX_CEILING,
        "lexical_similarity_threshold": SIMILARITY,
        "near_duplicate_policy": "QUARANTINE_ALL_CROSS_SPLIT_CLUSTERS",
        "candidate_tokenizable_examples": len(records),
        "near_duplicate_cluster_count": len(groups),
        "eligible_probe_indices": len(eligible),
        "independent_cluster_counts": dict(sorted(decisions.items())),
        "candidates": candidates,
        "raw_source_text_in_output": False,
        "near_duplicate_semantics_verified": False,
        "cross_dataset_contamination_verified": False,
        "independent_split_acceptance": False,
        "classifier_training_authorized": False,
        "production_authorized": False,
    }


def scan_existing_corpus(train_file: Path, model_dir: Path,
                         episode_key_file: Path):
    from transformers import AutoTokenizer
    from .k2_matched_compare import digest
    key = read_private_key(episode_key_file)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    entries, source = read_pairs(train_file)
    records = []
    for pairhash, (_, pair) in sorted(entries.items()):
        sample = tokenize_pair(tokenizer, pair, MAX_TOKENS)
        if sample is None:
            continue
        episode = pseudonym(key, sample)
        first_scored = sample["labels"].index(next(
            label for label in sample["labels"] if label != -100))
        # Scope: normalized source text only in this process; never persist.
        records.append({
            "sample_index": len(records),
            "normalized_prompt": normalized(pair[0]),
            "episode_hmac_sha256": episode,
            "partition": partition(episode),
            "prompt_token_count": first_scored,
            "assistant_token_count": sample["assistant_tokens"],
        })
        if len(records) >= MAX_CANDIDATES:
            break
    return build_manifest(
        records, source_sha256=source["sha256"],
        model_config_sha256=digest(model_dir / "config.json"))


def authorize_index(manifest: dict, *, source_sha256: str,
                    model_config_sha256: str, max_tokens: int,
                    sample_index: int, episode_hmac_sha256: str):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unrecognized preflight schema")
    if (manifest.get("source_dataset_sha256") != source_sha256
            or manifest.get("model_config_sha256") != model_config_sha256
            or manifest.get("max_tokens") != max_tokens):
        raise ValueError("manifest source, tokenizer or max-token drift")
    matches = [r for r in manifest.get("candidates", [])
               if r.get("sample_index") == sample_index]
    if len(matches) != 1 or matches[0]["disposition"] != "ELIGIBLE":
        raise ValueError("source index not individually approved by preflight")
    if matches[0]["episode_hmac_sha256"] != episode_hmac_sha256:
        raise ValueError("episode HMAC changed after preflight")
    return matches[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-preflight", action="store_true")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--episode-key-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.read_only_preflight:
        p.error("explicit --read-only-preflight required")
    if args.output.exists() or args.output.is_symlink():
        p.error("refuse existing cohort evidence")
    os.umask(0o077)
    manifest = scan_existing_corpus(
        args.train_jsonl, args.model_dir, args.episode_key_file)
    with args.output.open("x", encoding="utf-8") as file:
        file.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        key: value for key, value in manifest.items()
        if key not in ("candidates",)
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

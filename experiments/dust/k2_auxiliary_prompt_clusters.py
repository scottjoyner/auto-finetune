"""Private exact + exhaustive lexical cluster audit of local K2 candidate prompts.

Only runs on an already-present local JSONL with a pinned local tokenizer.
No prompts, tokens, paired responses, plain hashes, model outputs or gradients
are serialized. Opaque HMACs stay in mode-600 evidence on local SSD.

Unlike approximate neighbor indexing, this implementation enumerates EVERY
unordered distinct normalized-prompt pair surviving a mathematically safe
length upper bound, and only skips pairs when difflib.quick_ratio() proves
ratio() < 0.85. This gives exact *SequenceMatcher >= .85* connected
components, not semantic/paraphrase independence and not data-use approval.

At most 5000 exact-unique, tokenizable prompts / 12_497_500 pairs. A
monotonic wall-clock cutoff raises and DOES NOT publish a partial manifest.
No classifier/training/sampling authority is ever granted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import hmac
import json
import os
from pathlib import Path
import time

from .k2_data import read_pairs, tokenize_pair, normalized
from .k2_direction_witness import read_private_key
from .predictive_probe_contract import partition

SCHEMA = "auto-finetune.dust-k2-private-auxiliary-clusters.v1"
THRESHOLD = 0.85
MAX_DISTINCT = 5000
MAX_PAIRS = MAX_DISTINCT * (MAX_DISTINCT - 1) // 2
MAX_SECONDS = 240
MAX_TOKENS = 128


def digest_keyed(key: bytes, tag: str, data: str) -> str:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("private grouping key missing")
    return hmac.new(
        key, (tag + ":" + data).encode(), hashlib.sha256
    ).hexdigest()


def lexical_components(prompts: list[str], *, cutoff_seconds: int = MAX_SECONDS,
                       threshold: float = THRESHOLD):
    """Return exhaustive lexical connected components and CPU audit counters.

    Sorted-by-length pruning uses 2*len(shorter)/(len1+len2) >= threshold
    as a necessary upper bound for the SequenceMatcher match ratio. Any
    quick_ratio() < threshold is also a mathematically safe rejection.
    """
    if type(cutoff_seconds) is not int or not 1 <= cutoff_seconds <= MAX_SECONDS:
        raise ValueError("bounded deadline required")
    if type(threshold) is not float or threshold != THRESHOLD:
        raise ValueError("frozen lexical threshold 0.85 required")
    if not 1 <= len(prompts) <= MAX_DISTINCT:
        raise ValueError("candidate count outside exhaustive audit bound")
    if len(set(prompts)) != len(prompts):
        raise ValueError("exact duplicate prompts must be collapsed first")
    if any(not isinstance(s, str) or not s or len(s) > 4000 for s in prompts):
        raise ValueError("invalid private source prompt")
    order = sorted(range(len(prompts)), key=lambda x: (len(prompts[x]), x))
    parent = list(range(len(prompts)))
    comparisons = 0
    length_pruned = 0
    quick_pruned = 0
    edges = 0
    start = time.monotonic()

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ordinal, i in enumerate(order):
        left = prompts[i]
        n = len(left)
        for offset, j in enumerate(order[ordinal + 1:]):
            if (comparisons + length_pruned) % 32768 == 0:
                if time.monotonic() - start > cutoff_seconds:
                    raise TimeoutError("exhaustive lexical audit deadline: no manifest")
            right = prompts[j]
            m = len(right)
            # Increasing length: every remaining partner is also impossible.
            if 2 * n < threshold * (n + m):
                length_pruned += len(order) - (ordinal + 1) - offset
                break
            comparisons += 1
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            if matcher.quick_ratio() < threshold:
                quick_pruned += 1
                continue
            if matcher.ratio() >= threshold:
                edges += 1
                parent[root(i)] = root(j)
        if time.monotonic() - start > cutoff_seconds:
            raise TimeoutError("exhaustive lexical audit deadline: no manifest")
    clusters = defaultdict(list)
    for i in range(len(prompts)):
        clusters[root(i)].append(i)
    members = sorted(
        (tuple(sorted(c)) for c in clusters.values()),
        key=lambda c: (c[0], len(c)))
    return members, {
        "all_unordered_distinct_prompt_pairs": len(prompts) * (len(prompts)-1)//2,
        "length_eligible_pairs_evaluated": comparisons,
        "length_pruned_pairs": length_pruned,
        "quick_ratio_below_threshold_pairs": quick_pruned,
        "candidate_pair_accounting_complete": (
            comparisons + length_pruned == len(prompts) * (len(prompts)-1)//2),
        "sequence_matcher_edges_at_or_above_threshold": edges,
        "lexical_components": len(members),
        "max_component_unique_prompts": max(map(len, members)),
        "duration_upper_bound_seconds": cutoff_seconds,
        "safe_length_bound_and_quick_ratio_used": True,
        "unexamined_pairs_due_to_approximate_index": 0,
    }


def build_private_manifest(prompt_rows: list[dict], key: bytes, *,
                           source_sha: str, config_sha: str,
                           cutoff_seconds: int = MAX_SECONDS):
    if not all(
        isinstance(v, str) and len(v) == 64 and
        all(c in "0123456789abcdef" for c in v)
        for v in (source_sha, config_sha)
    ):
        raise ValueError("pinned source and tokenizer config sha required")
    if len(prompt_rows) == 0:
        raise ValueError("zero usable source prompts")
    unique = {}
    exact_multiplicity = Counter()
    for row in prompt_rows:
        prompt = row["normalized_prompt"]
        exact_multiplicity[prompt] += 1
        if prompt not in unique:
            unique[prompt] = row
    prompts = sorted(unique)
    components, evidence = lexical_components(
        prompts, cutoff_seconds=cutoff_seconds)
    entries = []
    splits = Counter()
    for cluster in components:
        ids = [digest_keyed(key, "exact-prompt-v1", prompts[i]) for i in cluster]
        cluster_identity = digest_keyed(
            key, "lexical-component-v1", "|".join(sorted(ids)))
        split = partition(cluster_identity)
        splits[split] += 1
        # The cluster split is selected AFTER grouping all prompts. A prompt
        # and every response variant share exactly one split assignment.
        for i in cluster:
            prompt = prompts[i]
            entries.append({
                "prompt_hmac_sha256": digest_keyed(key, "exact-prompt-v1", prompt),
                "near_duplicate_cluster_hmac_sha256": cluster_identity,
                "partition_candidate_only": split,
                "paired_response_variants": exact_multiplicity[prompt],
                "sample_index": unique[prompt]["sample_index"],
                "prompt_token_count": unique[prompt]["prompt_token_count"],
                "assistant_token_count": unique[prompt]["assistant_token_count"],
                "training_eligible": False,
                "heldout_eligible": False,
            })
    entries.sort(key=lambda r: r["sample_index"])
    return {
        "schema": SCHEMA,
        "mode": "LOCAL_PRIVATE_EXHAUSTIVE_LEXICAL_CANDIDATE_AUDIT",
        "source_sha256": source_sha,
        "model_config_sha256": config_sha,
        "model_max_tokens": MAX_TOKENS,
        "lexical_threshold": THRESHOLD,
        "source_tokenizable_pair_rows": len(prompt_rows),
        "exact_unique_normalized_prompts": len(prompts),
        "response_variant_excess_rows": len(prompt_rows)-len(prompts),
        "max_response_variants_per_exact_prompt": max(exact_multiplicity.values()),
        "lexical_source_components": len(components),
        "candidate_partition_component_counts": {
            k: splits[k] for k in ("train", "validation", "test")
        },
        "lexical_audit": evidence,
        "candidate_entries": entries,
        "raw_prompt_response_or_token_data_exported": False,
        "private_hmac_identifiers_only": True,
        "semantic_paraphrase_independence_verified": False,
        "cross_corpus_semantic_contamination_verified": False,
        "dataset_reuse_rights_approved": False,
        "receiver_signing_key_independent": False,
        "source_partition_authorized": False,
        "classifier_training_authorized": False,
        "new_model_sampling_authorized": False,
        "optimizer_or_production_authorized": False,
    }


def read_local_source(source_path: Path, model_dir: Path, key_path: Path,
                      *, cutoff_seconds: int = MAX_SECONDS):
    from transformers import AutoTokenizer
    from .k2_matched_compare import digest
    key = read_private_key(key_path)
    source, info = read_pairs(source_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    rows = []
    for _, (_, pair) in sorted(source.items()):
        try:
            sample = tokenize_pair(tokenizer, pair, MAX_TOKENS)
        except (ValueError, KeyError, TypeError, AttributeError):
            sample = None
        if sample is None:
            continue
        first_scored = sample["labels"].index(next(
            label for label in sample["labels"] if label != -100))
        rows.append({
            "sample_index": len(rows),
            "normalized_prompt": normalized(pair[0]),
            "prompt_token_count": first_scored,
            "assistant_token_count": sample["assistant_tokens"],
        })
    result = build_private_manifest(
        rows, key, source_sha=info["sha256"],
        config_sha=digest(model_dir / "config.json"),
        cutoff_seconds=cutoff_seconds)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-private-cluster-audit", action="store_true")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--source-jsonl", required=True, type=Path)
    p.add_argument("--episode-key-file", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cutoff-seconds", type=int, default=MAX_SECONDS)
    args = p.parse_args(argv)
    if not args.read_only_private_cluster_audit:
        p.error("requires --read-only-private-cluster-audit")
    if args.output.exists() or args.output.is_symlink():
        p.error("refuse any existing evidence path")
    if not args.output.parent.is_dir() or args.output.parent.stat().st_mode & 0o077:
        p.error("private evidence must go in mode-700 directory")
    result = read_local_source(
        args.source_jsonl, args.model_dir, args.episode_key_file,
        cutoff_seconds=args.cutoff_seconds)
    os.umask(0o077)
    with args.output.open("x", encoding="utf-8") as fd:
        fd.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    # stdout aggregate only, no HMAC identities from candidate entries.
    print(json.dumps({
        k: v for k, v in result.items() if k != "candidate_entries"
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

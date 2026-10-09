"""Order-invariant token-set overlap triage between two local prompt corpora.

A cheap privacy-preserving complement to the existing exhaustive 0.85
character SequenceMatcher scan. Word-order changes may evade the original
rule; this scans exact normalized *sets* of Unicode word tokens at
Jaccard >=0.75 and at least four unique tokens on each side. This is NOT
semantic paraphrase detection, even if no token overlaps are found.

No raw prompts, token sets, prompt HMACs, or individual matched examples
are printed or persisted; output is counts and pinned source file SHA256.
Every source/aux pair is either scored or mathematically size-pruned.
Bounded runtime, pair count, source count and input rows; fail without
producing partial reports on timeout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .k2_data import read_pairs, normalized

SCHEMA = "auto-finetune.dust-k2-token-set-overlap.v1"
THRESHOLD = 0.75
MAX_ORIGINAL = 1024
MAX_AUXILIARY = 16384
MAX_SECONDS = 180
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def token_set(value: str) -> frozenset[str]:
    if not isinstance(value, str):
        raise ValueError("invalid prompt")
    return frozenset(TOKEN_PATTERN.findall(normalized(value)))


def compare_normalized_prompts(source: list[str], auxiliary: list[str], *,
                               source_sha256: str, auxiliary_sha256: str,
                               deadline_seconds: int = MAX_SECONDS):
    if (type(deadline_seconds) is not int or
            not 1 <= deadline_seconds <= MAX_SECONDS):
        raise ValueError("unbounded token overlap deadline")
    if (not 1 <= len(source) <= MAX_ORIGINAL
            or not 1 <= len(auxiliary) <= MAX_AUXILIARY):
        raise ValueError("unexpected original/auxiliary corpus size")
    for digest in (source_sha256, auxiliary_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(
                x not in "0123456789abcdef" for x in digest):
            raise ValueError("missing immutable corpus SHA256")
    originals = sorted(set(source))
    comparison = sorted(set(auxiliary))
    src_tokens = [token_set(x) for x in originals]
    aux_tokens = [token_set(x) for x in comparison]
    n_pairs = len(src_tokens) * len(aux_tokens)
    if n_pairs > MAX_ORIGINAL * MAX_AUXILIARY:
        raise ValueError("too many pairs")
    start = time.monotonic()
    scored, pruned = 0, 0
    matched_sources = set()
    matching_pairs = 0
    short_src = sum(len(t) < 4 for t in src_tokens)
    short_aux = sum(len(t) < 4 for t in aux_tokens)
    for i, a in enumerate(src_tokens):
        for j, b in enumerate(aux_tokens):
            if (j & 8191) == 0 and time.monotonic() - start > deadline_seconds:
                raise TimeoutError("token overlap scan exceeded deadline")
            minsize = min(len(a), len(b))
            maxsize = max(len(a), len(b))
            # Jaccard cannot exceed shorter_set / longer_set.
            if minsize < 4 or minsize < THRESHOLD * maxsize:
                pruned += 1
                continue
            scored += 1
            intersection = len(a & b)
            union = len(a) + len(b) - intersection
            if union and intersection / union >= THRESHOLD:
                matching_pairs += 1
                matched_sources.add(i)
    if n_pairs != scored + pruned:
        raise RuntimeError("token pair accounting incomplete")
    return {
        "schema": SCHEMA,
        "mode": "READ_ONLY_ORDER_INVARIANT_TOKEN_OVERLAP_TRIAGE",
        "source_sha256": source_sha256,
        "auxiliary_sha256": auxiliary_sha256,
        "token_jaccard_threshold": THRESHOLD,
        "min_unique_tokens_for_comparison": 4,
        "source_exact_distinct_prompts": len(originals),
        "auxiliary_exact_distinct_prompts": len(comparison),
        "source_prompts_too_short_for_token_triage": short_src,
        "auxiliary_prompts_too_short_for_token_triage": short_aux,
        "all_source_auxiliary_prompt_pairs": n_pairs,
        "size_pruned_pairs": pruned,
        "token_jaccard_scored_pairs": scored,
        "high_token_overlap_pairs": matching_pairs,
        "unique_original_prompts_with_high_overlap": len(matched_sources),
        "pair_accounting_complete": True,
        "semantic_paraphrase_detection_established": False,
        "syntactic_token_reorder_only": True,
        "dataset_rights_approved": False,
        "new_source_partitions_authorized": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
        "raw_prompt_token_sets_or_hmac_in_report": False,
    }


def run(source_path: Path, aux_path: Path, *,
        expected_source_sha: str, expected_aux_sha: str):
    if source_path.resolve() == aux_path.resolve():
        raise ValueError("cannot compare file with itself")
    originals, source_info = read_pairs(source_path)
    additional, aux_info = read_pairs(aux_path)
    if (source_info["sha256"] != expected_source_sha
            or aux_info["sha256"] != expected_aux_sha):
        raise ValueError("local source file drift from pinned SHA")
    return compare_normalized_prompts(
        [normalized(pair[0]) for _, (_, pair) in originals.items()],
        [normalized(pair[0]) for _, (_, pair) in additional.items()],
        source_sha256=expected_source_sha,
        auxiliary_sha256=expected_aux_sha)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-token-triage", action="store_true")
    p.add_argument("--source-jsonl", required=True, type=Path)
    p.add_argument("--auxiliary-jsonl", required=True, type=Path)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--expected-auxiliary-sha256", required=True)
    p.add_argument("--private-output", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.read_only_token_triage:
        p.error("explicit --read-only-token-triage required")
    target = args.private_output
    if (target.is_symlink() or target.exists()
            or not target.parent.is_dir() or target.parent.stat().st_mode & 0o077):
        p.error("must write a new private research report")
    result = run(
        args.source_jsonl, args.auxiliary_jsonl,
        expected_source_sha=args.expected_source_sha256,
        expected_aux_sha=args.expected_auxiliary_sha256)
    os.umask(0o077)
    with target.open("x", encoding="utf-8") as file:
        file.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        file.flush()
        os.fsync(file.fileno())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

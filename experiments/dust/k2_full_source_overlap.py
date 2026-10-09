"""Full original K2 training-source prompt overlap against a local auxiliary.

Unlike the 75-tokenizable preflight, this checks ALL normalized prompts in
the existing source-pair reader (up to 1024 pairs) against ALL paired
auxiliary prompts (up to 16384). Input source and auxiliary SHA256 must be
pinned. It reports aggregate exact/lexical matching, never prompt strings.
This is not a semantic duplicate screen or approval of a new dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .k2_data import read_pairs, normalized
from .k2_cross_corpus_audit import lexical_matches

SCHEMA = "auto-finetune.dust-k2-full-source-lexical-screen.v1"
SOURCE_LIMIT = 1024
AUX_LIMIT = 16384


def screen_full_source_prompts(source_prompts: list[str],
                               auxiliary_prompts: list[str], *,
                               source_sha: str, auxiliary_sha: str):
    if (
        not 1 <= len(source_prompts) <= SOURCE_LIMIT
        or not 1 <= len(auxiliary_prompts) <= AUX_LIMIT
        or any(not isinstance(s, str) or not s
               for s in source_prompts + auxiliary_prompts)
    ):
        raise ValueError("source/aux pair count outside bounded lexical audit")
    primary = sorted(set(source_prompts))
    other = sorted(set(auxiliary_prompts))
    rows = [
        {"sample_index": i, "normalized_prompt": text}
        for i, text in enumerate(primary)
    ]
    matches = lexical_matches(rows, other)
    exact = sum(m["exact"] for m in matches.values())
    for name, sha in (("source", source_sha), ("auxiliary", auxiliary_sha)):
        if (not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError(name + " SHA missing")
    return {
        "schema": SCHEMA,
        "mode": "LOCAL_READ_ONLY_ALL_ORIGINAL_PROMPT_LEXICAL_SCREEN",
        "source_sha256": source_sha,
        "auxiliary_sha256": auxiliary_sha,
        "source_unique_paired_rows": len(source_prompts),
        "source_exact_distinct_prompts": len(primary),
        "auxiliary_unique_paired_rows": len(auxiliary_prompts),
        "auxiliary_exact_distinct_prompts": len(other),
        "lexically_overlapping_original_prompt_count": len(matches),
        "exactly_matching_original_prompt_count": exact,
        "lexical_threshold": .85,
        "all_source_pairs_examined": True,
        "all_auxiliary_pairs_examined": True,
        "semantic_paraphrase_independence_verified": False,
        "dataset_rights_approved": False,
        "source_splits_authorized": False,
        "classifier_training_authorized": False,
        "production_authorized": False,
        "raw_prompts_or_tokens_emitted": False,
    }


def run(*, source: Path, auxiliary: Path, source_sha: str, auxiliary_sha: str):
    if source.resolve() == auxiliary.resolve():
        raise ValueError("source cannot also be its own auxiliary")
    src_entries, src_stats = read_pairs(source)
    aux_entries, aux_stats = read_pairs(auxiliary)
    if (src_stats["sha256"] != source_sha
            or aux_stats["sha256"] != auxiliary_sha):
        raise ValueError("source or auxiliary file changed from expected SHA256")
    originals = [normalized(pair[0]) for _, (_, pair)
                 in sorted(src_entries.items())]
    others = [normalized(pair[0]) for _, (_, pair)
              in sorted(aux_entries.items())]
    return screen_full_source_prompts(
        originals, others, source_sha=source_sha, auxiliary_sha=auxiliary_sha)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-all-source", action="store_true")
    p.add_argument("--source-jsonl", required=True, type=Path)
    p.add_argument("--auxiliary-jsonl", required=True, type=Path)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--auxiliary-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.read_only_all_source:
        p.error("explicit --read-only-all-source required")
    if (args.output.is_symlink() or args.output.exists()
            or not args.output.parent.is_dir()
            or args.output.parent.stat().st_mode & 0o077):
        p.error("new mode-600 audit path in mode-700 directory required")
    result = run(
        source=args.source_jsonl, auxiliary=args.auxiliary_jsonl,
        source_sha=args.source_sha256, auxiliary_sha=args.auxiliary_sha256)
    os.umask(0o077)
    with args.output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

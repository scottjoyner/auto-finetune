"""Independent aggregate-only verifier of private K2 auxiliary candidate manifest.

The cluster builder itself is producer-local and not a semantic witness.
This verifier checks exact source SHA, mode-600 custody, bounded JSON size,
no duplicate HMAC prompt identities, one split per related prompt family,
count and pair-accounting integrity and explicit training / rights HOLD.

Does not access prompts, token IDs, model weights, receiver signing keys or
GPU. Digest-matched local evidence is NOT permission to train or deploy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import stat

from .k2_auxiliary_prompt_clusters import SCHEMA, MAX_DISTINCT
from .predictive_probe_contract import partition

MAX_BYTES = 5 * 1024 * 1024
SPLITS = ("train", "validation", "test")


def require_hex(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("invalid " + name)
    return value


def verify_manifest(path: Path, *, exact_sha: str, source_sha: str,
                    config_sha: str) -> dict:
    for name, value in (
        ("private manifest SHA", exact_sha),
        ("source SHA", source_sha),
        ("tokenizer config SHA", config_sha),
    ):
        require_hex(value, name)
    if path.is_symlink() or not path.is_file():
        raise ValueError("source manifest is missing or a symlink")
    st = path.stat()
    if stat.S_IMODE(st.st_mode) & 0o077 or not 0 < st.st_size <= MAX_BYTES:
        raise PermissionError("manifest must be private mode 600 and bounded")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != exact_sha:
        raise ValueError("pinned private manifest SHA256 changed")
    def pairs_no_duplicates(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicated manifest JSON key")
            result[key] = value
        return result
    manifest = json.loads(
        raw, object_pairs_hook=pairs_no_duplicates,
        parse_constant=lambda v: (_ for _ in ()).throw(
            ValueError("nonfinite manifest value")))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("mode") !=
            "LOCAL_PRIVATE_EXHAUSTIVE_LEXICAL_CANDIDATE_AUDIT"
        or manifest.get("source_sha256") != source_sha
        or manifest.get("model_config_sha256") != config_sha
        or manifest.get("model_max_tokens") != 128
        or manifest.get("lexical_threshold") != .85
    ):
        raise ValueError("source, model or experiment schema drift")
    for name, required in (
        ("classifier_training_authorized", False),
        ("source_partition_authorized", False),
        ("dataset_reuse_rights_approved", False),
        ("receiver_signing_key_independent", False),
        ("semantic_paraphrase_independence_verified", False),
        ("cross_corpus_semantic_contamination_verified", False),
        ("new_model_sampling_authorized", False),
        ("optimizer_or_production_authorized", False),
        ("raw_prompt_response_or_token_data_exported", False),
        ("private_hmac_identifiers_only", True),
    ):
        if manifest.get(name) is not required:
            raise ValueError("unexpected data promotion or privacy claim: " + name)
    entries = manifest.get("candidate_entries")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_DISTINCT:
        raise ValueError("invalid private prompt candidate count")
    seen_prompt = set()
    seen_sample = set()
    family_splits = {}
    sizes = Counter()
    counted_splits = defaultdict(set)
    variant_sum = 0
    max_variants = 0
    fields = frozenset({
        "prompt_hmac_sha256", "near_duplicate_cluster_hmac_sha256",
        "partition_candidate_only", "paired_response_variants",
        "sample_index", "prompt_token_count", "assistant_token_count",
        "training_eligible", "heldout_eligible",
    })
    for row in entries:
        if not isinstance(row, dict) or frozenset(row) != fields:
            raise ValueError("unexpected private candidate field")
        prompt = require_hex(row["prompt_hmac_sha256"], "prompt HMAC")
        family = require_hex(row["near_duplicate_cluster_hmac_sha256"],
                             "cluster HMAC")
        i = row["sample_index"]
        if (type(i) is not int or i < 0
                or i in seen_sample or prompt in seen_prompt):
            raise ValueError("replayed prompt or source sample index")
        seen_prompt.add(prompt)
        seen_sample.add(i)
        split = row["partition_candidate_only"]
        if split not in SPLITS or split != partition(family):
            raise ValueError("family split disagrees with frozen HMAC rule")
        if family in family_splits and family_splits[family] != split:
            raise ValueError("one near-duplicate family crosses provisional split")
        family_splits[family] = split
        sizes[family] += 1
        counted_splits[split].add(family)
        count = row["paired_response_variants"]
        if (type(count) is not int or not 1 <= count <= 11037
                or type(row["prompt_token_count"]) is not int
                or type(row["assistant_token_count"]) is not int
                or not 0 < row["prompt_token_count"] <= 128
                or not 0 < row["assistant_token_count"] <= 128
                or row["training_eligible"] is not False
                or row["heldout_eligible"] is not False):
            raise ValueError("invalid source row counts or training authorization")
        variant_sum += count
        max_variants = max(max_variants, count)
    pair_stats = manifest.get("lexical_audit", {})
    distinct_pairs = len(entries) * (len(entries) - 1) // 2
    if (
        manifest.get("source_tokenizable_pair_rows") != variant_sum
        or manifest.get("exact_unique_normalized_prompts") != len(entries)
        or manifest.get("response_variant_excess_rows") != variant_sum-len(entries)
        or manifest.get("max_response_variants_per_exact_prompt") != max_variants
        or manifest.get("lexical_source_components") != len(sizes)
        or manifest.get("candidate_partition_component_counts") != {
            s: len(counted_splits[s]) for s in SPLITS}
        or pair_stats.get("all_unordered_distinct_prompt_pairs") != distinct_pairs
        or pair_stats.get("candidate_pair_accounting_complete") is not True
        or pair_stats.get("unexamined_pairs_due_to_approximate_index") != 0
        or pair_stats.get("lexical_components") != len(sizes)
        or pair_stats.get("max_component_unique_prompts") != max(sizes.values())
        or pair_stats.get("length_eligible_pairs_evaluated", -1)
            + pair_stats.get("length_pruned_pairs", -1) != distinct_pairs
        or type(pair_stats.get("sequence_matcher_edges_at_or_above_threshold"))
            is not int
        or not 0 <= pair_stats["sequence_matcher_edges_at_or_above_threshold"]
            <= pair_stats["length_eligible_pairs_evaluated"]
    ):
        raise ValueError("manifest source counts or exhaustive pair proof inconsistent")
    return {
        "schema": "auto-finetune.dust-k2-private-cluster-reconciliation.v1",
        "result": "PASS_PRIVATE_LEXICAL_ACCOUNTING_ONLY",
        "source_exact_sha256": source_sha,
        "manifest_exact_sha256": exact_sha,
        "tokenizable_paired_rows": variant_sum,
        "unique_exact_prompts": len(entries),
        "lexical_components": len(sizes),
        "max_distinct_prompts_in_component": max(sizes.values()),
        "source_candidate_component_counts_by_split": {
            s: len(counted_splits[s]) for s in SPLITS},
        "unexamined_approximate_index_pairs": 0,
        "semantic_independence_verified": False,
        "dataset_reuse_rights_approved": False,
        "receiver_key_custody_independent": False,
        "train_validation_test_split_authorized": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
        "private_identifiers_or_prompts_in_report": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--expected-manifest-sha256", required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--expected-model-config-sha256", required=True)
    args = p.parse_args(argv)
    if not args.verify_only:
        p.error("explicit --verify-only required")
    result = verify_manifest(
        args.manifest, exact_sha=args.expected_manifest_sha256,
        source_sha=args.expected_source_sha256,
        config_sha=args.expected_model_config_sha256)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

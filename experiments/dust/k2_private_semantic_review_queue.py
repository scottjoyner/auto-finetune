"""Private, immutable human-review queue for MiniLM cross-split neighbors.

Prerequisite: exact private PR #31 cluster manifest plus PR #33 aggregate
semantic report. Repeats frozen local CPU MiniLM inference over the same
4,142 *K2-tokenizable* auxiliary prompts and checks the pair counts
against the *existing SHA-pinned aggregate* BEFORE queue export.

Queue entries are source prompt HMAC IDs, related-family HMAC IDs, split
labels and cosine score ONLY. No raw prompts, responses, token IDs,
embeddings, access keys or trained weights are persisted or sent to GitHub.

Crucially these 0.85 cosine edges are REVIEW CANDIDATES, not certified
duplicate labels. Failed frozen detector challenge means that edges missed
by this model can still contain semantic duplicates. No split, classifier
or optimizer operation is authorized by this queue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from .k2_auxiliary_cluster_reconcile import verify_manifest
from .k2_auxiliary_prompt_clusters import digest_keyed
from .k2_data import normalized, read_pairs, tokenize_pair
from .k2_direction_witness import read_private_key
from .k2_matched_compare import digest
from .k2_offline_semantic_audit import encode, full_sha

SCHEMA = "auto-finetune.dust-k2-private-semantic-review-queue.v1"
THRESHOLD = .85
MAX_QUEUE = 500
MAX_SOURCE = 5000
EMBED_BATCH = 128
REQUIRED_AGGREGATE_SCHEMA = "auto-finetune.dust-k2-offline-semantic-triage.v1"


def validate_private_json(path: Path, expected_sha: str, *, limit=5_000_000):
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha)):
        raise ValueError("expected immutable report SHA256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("unsafe or missing pinned evidence file")
    st = path.stat()
    if stat.S_IMODE(st.st_mode) & 0o077 or not 0 < st.st_size <= limit:
        raise PermissionError("read only private bounded evidence")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("existing audit SHA256 changed")
    return json.loads(data)


def make_review_queue(*, pairs, ids, groups, splits,
                      source_sha, cluster_sha, semantic_sha, model_sha):
    if (len(ids) != len(groups) or len(ids) != len(splits)
            or len(set(ids)) != len(ids)
            or len(ids) > MAX_SOURCE
            or not 1 <= len(ids)):
        raise ValueError("invalid or duplicated prompt identifiers")
    if any(s not in ("train", "validation", "test") for s in splits):
        raise ValueError("invalid source family partition")
    for hash_id in (source_sha, cluster_sha, semantic_sha, model_sha, *ids, *groups):
        if (not isinstance(hash_id, str) or len(hash_id) != 64
                or any(c not in "0123456789abcdef" for c in hash_id)):
            raise ValueError("invalid immutable identity")
    queue = []
    seen = set()
    all_cross_family = 0
    for i, j, similarity in pairs:
        if (type(i) is not int or type(j) is not int
                or not 0 <= i < j < len(ids)
                or type(similarity) not in (int, float)
                or not math.isfinite(similarity)
                or not THRESHOLD <= similarity <= 1.00001
                or (i, j) in seen):
            raise ValueError("forged or repeated semantic neighbor index")
        seen.add((i, j))
        if groups[i] == groups[j]:
            if splits[i] != splits[j]:
                raise ValueError("existing lexical family crosses split")
            continue
        all_cross_family += 1
        if splits[i] == splits[j]:
            continue
        pair_id = hashlib.sha256(
            ("k2-private-review-v1:" + "|".join(sorted((ids[i], ids[j])))
             + ":" + model_sha).encode()).hexdigest()
        queue.append({
            "pair_id_sha256": pair_id,
            "left_prompt_hmac_sha256": ids[i],
            "right_prompt_hmac_sha256": ids[j],
            "left_family_hmac_sha256": groups[i],
            "right_family_hmac_sha256": groups[j],
            "left_candidate_partition": splits[i],
            "right_candidate_partition": splits[j],
            "cosine_similarity": round(float(similarity), 7),
            "human_label": "UNREVIEWED",
            "automatic_duplicate_certification": False,
            "independent_review_required": True,
        })
    if len(queue) > MAX_QUEUE:
        raise ValueError("review cap exceeded; no partial queue")
    queue.sort(key=lambda row: (-row["cosine_similarity"], row["pair_id_sha256"]))
    if len({x["pair_id_sha256"] for x in queue}) != len(queue):
        raise ValueError("HMAC review pair collision")
    return {
        "schema": SCHEMA,
        "policy": "PRIVATE_HUMAN_REVIEW_ONLY__NO_AUTOMATIC_QUARANTINE",
        "source_sha256": source_sha,
        "cluster_manifest_sha256": cluster_sha,
        "original_semantic_audit_sha256": semantic_sha,
        "encoder_weights_sha256": model_sha,
        "frozen_cosine_threshold": THRESHOLD,
        "all_candidate_cross_family_edges": all_cross_family,
        "cross_partition_review_candidates": len(queue),
        "review_candidates": queue,
        "source_prompt_text_or_embeddings_present": False,
        "reviewer_identified_or_adjudicated": False,
        "human_decisions_observed": 0,
        "semantic_independence_certified": False,
        "dataset_rights_approved": False,
        "receipt_signer_key_isolated": False,
        "train_validation_test_split_authorized": False,
        "classifier_training_authorized": False,
        "production_or_optimizer_authorized": False,
    }


def produce_queue(*, source_path: Path, source_sha: str,
                  family_path: Path, family_sha: str,
                  prior_audit_path: Path, prior_audit_sha: str,
                  source_key_path: Path, k2_model_dir: Path,
                  k2_config_sha: str, encoder_dir: Path):
    import numpy as np
    original = validate_private_json(
        prior_audit_path, prior_audit_sha, limit=100_000)
    if (original.get("schema") != REQUIRED_AGGREGATE_SCHEMA or
        original["threshold_summaries"]["0.85"][
            "all_embedding_neighbours_require_review"] is not True or
        original["auxiliary_source_sha256"] != source_sha or
        original["private_source_family_manifest_sha256"] != family_sha):
        raise ValueError("prior semantic evidence not a valid frozen reference")
    verify_manifest(
        family_path, exact_sha=family_sha, source_sha=source_sha,
        config_sha=k2_config_sha)
    manifest = validate_private_json(
        family_path, family_sha, limit=5_000_000)
    if digest(k2_model_dir / "config.json") != k2_config_sha:
        raise ValueError("K2 config revision drift")
    entries, stats = read_pairs(source_path)
    if stats["sha256"] != source_sha:
        raise ValueError("auxiliary source bytes changed")
    key = read_private_key(source_key_path)
    hmap = {
        r["prompt_hmac_sha256"]: r
        for r in manifest["candidate_entries"]
    }
    if len(hmap) != manifest["exact_unique_normalized_prompts"]:
        raise ValueError("source HMAC map lost uniqueness")
    from transformers import AutoTokenizer
    k2_tokenizer = AutoTokenizer.from_pretrained(
        str(k2_model_dir), local_files_only=True, trust_remote_code=True)
    qualified = set()
    for _, (_, pair) in sorted(entries.items()):
        try:
            okay = tokenize_pair(k2_tokenizer, pair, 128)
        except (ValueError, TypeError, AttributeError, KeyError):
            okay = None
        if okay is not None:
            qualified.add(normalized(pair[0]))
    prompts = sorted(qualified)
    if len(prompts) != original["auxiliary_tokenizable_unique_prompt_count"]:
        raise ValueError("tokenizable cohort no longer equals prior audit")
    identifiers = [
        digest_keyed(key, "exact-prompt-v1", prompt) for prompt in prompts]
    if any(x not in hmap for x in identifiers):
        raise ValueError("private prompt HMAC not found in source-family manifest")
    groups = [hmap[x]["near_duplicate_cluster_hmac_sha256"]
              for x in identifiers]
    splits = [hmap[x]["partition_candidate_only"]
              for x in identifiers]
    if encoder_dir.is_symlink() or not encoder_dir.is_dir():
        raise ValueError("encoder must be local frozen snapshot")
    weights = encoder_dir / "model.safetensors"
    if not weights.is_file():
        raise ValueError("missing local encoder weights")
    weight_sha = full_sha(weights)
    if weight_sha != original["embedding_model_weights_sha256"]:
        raise ValueError("embedding model changed from prior audit")
    embeddings = encode(encoder_dir, prompts)
    matches = []
    for begin in range(0, len(prompts), EMBED_BATCH):
        cosine = embeddings[begin:begin+EMBED_BATCH] @ embeddings.T
        rows, cols = np.nonzero(cosine >= THRESHOLD)
        matches.extend(
            (begin+int(i), int(j), float(cosine[i, j]))
            for i, j in zip(rows, cols)
            if begin+int(i) < int(j))
    queue = make_review_queue(
        pairs=matches, ids=identifiers, groups=groups, splits=splits,
        source_sha=source_sha, cluster_sha=family_sha,
        semantic_sha=prior_audit_sha, model_sha=weight_sha)
    expected = original["threshold_summaries"]["0.85"]
    if (queue["all_candidate_cross_family_edges"] !=
            expected["within_auxiliary_new_cross_family_edges"]
            or queue["cross_partition_review_candidates"] !=
            expected["within_auxiliary_new_cross_split_edges"]):
        raise ValueError("review candidate count disagrees with frozen PR33 audit")
    queue["prior_aggregate_reconciled"] = True
    return queue


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-review-queue-only", action="store_true")
    p.add_argument("--source-jsonl", required=True, type=Path)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--cluster-manifest", required=True, type=Path)
    p.add_argument("--cluster-sha256", required=True)
    p.add_argument("--prior-semantic-audit", required=True, type=Path)
    p.add_argument("--prior-semantic-audit-sha256", required=True)
    p.add_argument("--episode-key-file", required=True, type=Path)
    p.add_argument("--k2-model-dir", required=True, type=Path)
    p.add_argument("--k2-config-sha256", required=True)
    p.add_argument("--encoder-dir", required=True, type=Path)
    p.add_argument("--private-output", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.private_review_queue_only:
        p.error("explicit private review queue mode required")
    dest = args.private_output
    if (dest.exists() or dest.is_symlink() or
            not dest.parent.is_dir()
            or dest.parent.stat().st_mode & 0o077):
        p.error("exclusive private output path required")
    report = produce_queue(
        source_path=args.source_jsonl, source_sha=args.source_sha256,
        family_path=args.cluster_manifest, family_sha=args.cluster_sha256,
        prior_audit_path=args.prior_semantic_audit,
        prior_audit_sha=args.prior_semantic_audit_sha256,
        source_key_path=args.episode_key_file,
        k2_model_dir=args.k2_model_dir, k2_config_sha=args.k2_config_sha256,
        encoder_dir=args.encoder_dir)
    os.umask(0o077)
    with dest.open("x", encoding="utf-8") as fd:
        fd.write(json.dumps(report, sort_keys=True, indent=2)+"\n")
        fd.flush()
        os.fsync(fd.fileno())
    # Strict aggregate-only stdout. Never log pair or HMAC identifiers.
    print(json.dumps({
        k:v for k,v in report.items() if k != "review_candidates"
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

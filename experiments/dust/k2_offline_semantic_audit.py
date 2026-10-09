"""Offline MiniLM cosine-candidate audit for K2 prompt-family research.

Uses an already cached frozen sentence embedding transformer locally, no
sentence-transformers dependency or network access. Pure forward inference,
no gradients, no adapter or optimizer, no K2 forward execution.

The frozen 20-pair semantic challenge is scored BEFORE real corpora. Every
result remains a *review candidate*, never semantic ground truth.
Original prompt strings, auxiliary prompt strings and embeddings remain
in this Python process only; only aggregate counts and immutable source/model
digests are written to a new mode-600 SSD JSON evidence file.

Requires existing private family HMAC manifest from PR #31 and matches
every eligible prompt against it. Hard cap 5000 candidate prompts and
1024 source prompts; no quota, NAS, external calls or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from .k2_auxiliary_prompt_clusters import digest_keyed
from .k2_auxiliary_cluster_reconcile import verify_manifest
from .k2_data import normalized, read_pairs, tokenize_pair
from .k2_direction_witness import read_private_key
from .k2_matched_compare import digest
from .k2_semantic_challenge_contract import (
    PAIRS, THRESHOLDS, SCHEMA as CHALLENGE_SCHEMA,
    evaluate_challenge, summarize_collisions,
)

SCHEMA = "auto-finetune.dust-k2-offline-semantic-triage.v1"
MAX_AUX = 5000
MAX_SOURCE = 1024
EMBED_BATCH = 48
TOKEN_MAX = 128


def full_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def encode(model_dir: Path, strings: list[str]):
    """Frozen local CPU inference. No hosted inference/network fallback."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not strings or len(strings) > MAX_AUX + MAX_SOURCE + 40:
        raise ValueError("unbounded embedding task")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        str(model_dir), local_files_only=True,
        trust_remote_code=False).to("cpu").eval()
    if any(x.requires_grad for x in model.parameters()):
        model.requires_grad_(False)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(strings), EMBED_BATCH):
            tokens = tokenizer(
                strings[start:start + EMBED_BATCH],
                padding=True, truncation=True, max_length=TOKEN_MAX,
                return_tensors="pt")
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def review_embeddings(
    model_dir: Path, source: list[str], candidates: list[str],
    family_groups: list[str], family_splits: list[str]
) -> dict:
    import numpy as np

    if not 1 <= len(source) <= MAX_SOURCE or not 1 <= len(candidates) <= MAX_AUX:
        raise ValueError("unbounded source/candidate population")
    if len(candidates) != len(family_groups) or len(candidates) != len(family_splits):
        raise ValueError("family mapping lost")
    challenge_strings = [
        text for first, second, _ in PAIRS for text in (first, second)]
    all_embeddings = encode(
        model_dir, challenge_strings + source + candidates)
    n_challenge = len(challenge_strings)
    challenge_scores = [
        float(np.dot(all_embeddings[i], all_embeddings[i+1]))
        for i in range(0, n_challenge, 2)
    ]
    challenge = evaluate_challenge(challenge_scores)
    src = all_embeddings[n_challenge:n_challenge+len(source)]
    aux = all_embeddings[n_challenge+len(source):]
    cross_hits = {t: [] for t in THRESHOLDS}
    within_hits = {t: [] for t in THRESHOLDS}
    # Matrix-block implementation: no approximate nearest neighbors.
    # Captured evidence is derived summary only, never actual vectors.
    for begin in range(0, len(source), 128):
        sims = np.matmul(src[begin:begin+128], aux.T)
        for t in THRESHOLDS:
            ii, jj = np.nonzero(sims >= t)
            cross_hits[t].extend(
                (begin+int(i), int(j), float(sims[i, j]))
                for i, j in zip(ii, jj))
    for begin in range(0, len(candidates), 128):
        sims = np.matmul(aux[begin:begin+128], aux.T)
        for t in THRESHOLDS:
            ii, jj = np.nonzero(sims >= t)
            within_hits[t].extend(
                (begin+int(i), int(j), float(sims[i, j]))
                for i, j in zip(ii, jj)
                if begin+int(i) < int(j))
    return {
        "challenge": challenge,
        "threshold_summaries": {
            str(t): summarize_collisions(
                source_groups=["original"] * len(source),
                auxiliary_groups=family_groups,
                auxiliary_splits=family_splits,
                cross_hits=cross_hits[t], within_hits=within_hits[t],
                threshold=t)
            for t in THRESHOLDS
        },
        "fully_evaluated_cross_embedding_pairs":
            len(source)*len(candidates),
        "fully_evaluated_unordered_aux_embedding_pairs":
            len(candidates)*(len(candidates)-1)//2,
        "all_candidate_pairs_scored_not_nearest_neighbor_approximated": True,
    }


def run(*, source_path: Path, auxiliary_path: Path,
        source_sha: str, auxiliary_sha: str, family_path: Path,
        family_sha: str, source_key_path: Path, k2_model_dir: Path,
        k2_config_sha: str, embedding_model_dir: Path):
    src, src_meta = read_pairs(source_path)
    aux, aux_meta = read_pairs(auxiliary_path)
    if src_meta["sha256"] != source_sha or aux_meta["sha256"] != auxiliary_sha:
        raise ValueError("source file SHA256 mismatch")
    verify_manifest(
        family_path, exact_sha=family_sha, source_sha=auxiliary_sha,
        config_sha=k2_config_sha)
    family_manifest = json.loads(family_path.read_text())
    key = read_private_key(source_key_path)
    family_map = {
        row["prompt_hmac_sha256"]: (
            row["near_duplicate_cluster_hmac_sha256"],
            row["partition_candidate_only"])
        for row in family_manifest["candidate_entries"]
    }
    if len(family_map) != family_manifest["exact_unique_normalized_prompts"]:
        raise ValueError("private candidate family HMAC collision")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(k2_model_dir), local_files_only=True, trust_remote_code=True)
    aux_valid = set()
    for _, (_, pair) in sorted(aux.items()):
        try:
            good = tokenize_pair(tokenizer, pair, 128)
        except (ValueError, TypeError, AttributeError, KeyError):
            good = None
        if good is not None:
            aux_valid.add(normalized(pair[0]))
    if len(aux_valid) != len(family_map):
        raise ValueError("K2-tokenizable source prompt count drift")
    candidates = sorted(aux_valid)
    groups, splits = [], []
    for prompt in candidates:
        pseudonym = digest_keyed(key, "exact-prompt-v1", prompt)
        if pseudonym not in family_map:
            raise ValueError("private corpus HMAC does not reconcile")
        group, split = family_map[pseudonym]
        groups.append(group)
        splits.append(split)
    originals = sorted({normalized(pair[0]) for _, (_, pair) in src.items()})
    if not 1 <= len(originals) <= MAX_SOURCE:
        raise ValueError("original prompt population out of bound")
    if digest(k2_model_dir / "config.json") != k2_config_sha:
        raise ValueError("K2 tokenizer pinned model config drift")
    if embedding_model_dir.is_symlink() or not embedding_model_dir.is_dir():
        raise ValueError("invalid local embedding snapshot")
    model_weights = embedding_model_dir / "model.safetensors"
    model_config = embedding_model_dir / "config.json"
    if not model_weights.is_file() or not model_config.is_file():
        raise ValueError("offline encoder snapshot incomplete")
    model_sha = full_sha(model_weights)
    model_cfg = full_sha(model_config)
    score = review_embeddings(
        embedding_model_dir, originals, candidates, groups, splits)
    return {
        "schema": SCHEMA,
        "mode": "LOCAL_CPU_EMBEDDING_CANDIDATE_REVIEW_ONLY",
        "embedding_model_family": "sentence-transformers/all-MiniLM-L12-v2",
        "embedding_model_weights_sha256": model_sha,
        "embedding_model_config_sha256": model_cfg,
        "tokenization_max_length": TOKEN_MAX,
        "original_source_sha256": source_sha,
        "auxiliary_source_sha256": auxiliary_sha,
        "private_source_family_manifest_sha256": family_sha,
        "k2_model_config_sha256": k2_config_sha,
        "source_unique_prompt_count": len(originals),
        "auxiliary_tokenizable_unique_prompt_count": len(candidates),
        **score,
        "real_prompt_text_or_vectors_in_evidence": False,
        "embedding_scores_are_candidates_not_semantic_ground_truth": True,
        "cross_corpus_independence_certified": False,
        "semantic_challenge_is_not_human_validation": True,
        "dataset_reuse_rights_approved": False,
        "commercial_training_approved": False,
        "receiver_key_custody_independent": False,
        "heldout_splits_authorized": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-local-semantic-triage", action="store_true")
    p.add_argument("--source-jsonl", type=Path, required=True)
    p.add_argument("--auxiliary-jsonl", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--auxiliary-sha256", required=True)
    p.add_argument("--private-cluster-manifest", required=True, type=Path)
    p.add_argument("--cluster-sha256", required=True)
    p.add_argument("--episode-key-file", required=True, type=Path)
    p.add_argument("--k2-model-dir", required=True, type=Path)
    p.add_argument("--k2-config-sha256", required=True)
    p.add_argument("--embedding-model-dir", required=True, type=Path)
    p.add_argument("--private-output", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.read_only_local_semantic_triage:
        p.error("explicit read-only triage opt-in required")
    target = args.private_output
    if (target.is_symlink() or target.exists() or
            not target.parent.is_dir() or target.parent.stat().st_mode & 0o077):
        p.error("new evidence file in private mode-700 directory required")
    os.umask(0o077)
    result = run(
        source_path=args.source_jsonl,
        auxiliary_path=args.auxiliary_jsonl,
        source_sha=args.source_sha256, auxiliary_sha=args.auxiliary_sha256,
        family_path=args.private_cluster_manifest, family_sha=args.cluster_sha256,
        source_key_path=args.episode_key_file,
        k2_model_dir=args.k2_model_dir, k2_config_sha=args.k2_config_sha256,
        embedding_model_dir=args.embedding_model_dir)
    with target.open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

"""Read-only source-lineage triage across SHA-pinned lexical audit files.

Purpose: Never treat a mixed/repacked source as novel heldout episodes.
No corpus is declared independent solely because its lexical overlap is
zero. Pinned lexical reports establish source *observations* and lineage
suspicions only. Licensing, semantic duplication, target leakage, tokenizer
eligibility, split authority and independent custody remain separate HOLDs.

No raw prompt strings, tokens, HMACs, per-direction features or response
text are exposed. Reports are aggregated and do not authorize collection,
classifier fitting, K2 optimization or production changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .k2_cohort_preflight import SCHEMA as COHORT_SCHEMA
from .k2_crosscorpus_veto import bind_quarantine_overlay
from .k2_feature16_readiness import (
    hex_digest, private_file, load_json,
)

SCHEMA = "auto-finetune.dust-k2-source-lineage-triage.v1"
MAX_AUDITS = 6


def classify_source(
    *, candidate_rows: int, exact_matches: int,
    lexical_matches: int, unique_pairs: int, screened: int,
    truncated: bool
) -> str:
    if (not all(type(x) is int for x in (
            candidate_rows, exact_matches, lexical_matches,
            unique_pairs, screened)) or
            not 1 <= candidate_rows <= 256 or
            not 0 <= exact_matches <= lexical_matches <= candidate_rows or
            not 0 <= screened <= unique_pairs):
        raise ValueError("invalid auxiliary source overlap accounting")
    if truncated is not (screened < unique_pairs):
        raise ValueError("auxiliary truncation status contradicts pair counts")
    if screened == 0:
        return "NO_COMPATIBLE_SOURCE_PAIRS"
    if exact_matches == candidate_rows:
        return "CONTAINS_ALL_CANDIDATE_PROMPTS_EXACTLY"
    if lexical_matches:
        return "SHARES_CANDIDATE_PROMPTS"
    if truncated:
        return "UNFINISHED_LEXICAL_SCAN"
    return "NO_DETECTED_LEXICAL_OVERLAP__NOT_INDEPENDENCE_PROOF"


def source_lineage_audit(
    manifest_path: Path, expected_manifest_sha: str,
    audits: list[tuple[Path, str]],
) -> dict:
    if not hex_digest(expected_manifest_sha):
        raise ValueError("no SHA pin for private source preflight")
    blob = private_file(manifest_path, cap=256 * 1024)
    if hashlib.sha256(blob).hexdigest() != expected_manifest_sha:
        raise ValueError("private cohort manifest hash mismatch")
    manifest = load_json(blob)
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError("unsupported source preflight format")
    if not 1 <= len(audits) <= MAX_AUDITS:
        raise ValueError("must pass 1..6 pinned corpus audits")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 256:
        raise ValueError("invalid source candidates")

    seen_shas = set()
    seen_names = set()
    outcomes = []
    observed_overlap_clust = set()
    complete_files = 0
    for audit_path, expected_sha in audits:
        overlay = bind_quarantine_overlay(
            manifest, expected_manifest_sha,
            audit_path, expected_sha)
        report = load_json(private_file(audit_path, cap=128 * 1024))
        observed_overlap_clust.update(overlay["quarantined_clusters"])
        for aux in report["aux_corpora"]:
            identity = (aux["filename"], aux["corpus_sha256"])
            if aux["filename"] in seen_names or aux["corpus_sha256"] in seen_shas:
                raise ValueError("auxiliary source counted more than once")
            seen_names.add(aux["filename"])
            seen_shas.add(aux["corpus_sha256"])
            status = classify_source(
                candidate_rows=len(candidates),
                exact_matches=aux["exactly_matched_source_indices"],
                lexical_matches=aux["overlapping_source_indices"],
                unique_pairs=aux["unique_pairs_reported"],
                screened=aux["pairs_screened"],
                truncated=aux["truncated"])
            is_complete = (
                aux["pairs_screened"] == aux["unique_pairs_reported"])
            complete_files += int(is_complete)
            outcomes.append({
                "filename": aux["filename"],
                "sha256": aux["corpus_sha256"],
                "reported_eligible_pair_count": aux["unique_pairs_reported"],
                "unique_pairs_screened": aux["pairs_screened"],
                "lexical_overlap_candidate_count":
                    aux["overlapping_source_indices"],
                "exact_overlap_candidate_count":
                    aux["exactly_matched_source_indices"],
                "coverage_complete": is_complete,
                "lineage_observation": status,
                "eligible_as_independent_heldout": False,
                "tokenizable_as_128_token_k2_candidates_proven": False,
                "legal_reuse_approved": False,
            })
    statuses = Counter(x["lineage_observation"] for x in outcomes)
    return {
        "schema": SCHEMA,
        "mode": "READ_ONLY_PINNED_SOURCE_LINEAGE",
        "source_cohort_sha256": expected_manifest_sha,
        "source_candidate_rows": len(candidates),
        "audited_corpora": len(outcomes),
        "full_lexical_corpus_scans": complete_files,
        "lexically_overlapping_existing_clusters": len(observed_overlap_clust),
        "source_lineage_observation_counts": dict(sorted(statuses.items())),
        "sources": outcomes,
        "source_independence_certified": False,
        "holdout_splits_approved": False,
        "all_known_corpora_exhaustively_scanned": False,
        "semantic_paraphrase_independence_verified": False,
        "dataset_rights_verified": False,
        "receiver_key_custody_isolated": False,
        "classifier_training_authorized": False,
        "new_model_sampling_authorized": False,
        "optimizer_or_production_authorized": False,
        "contains_user_prompt_text_or_episode_hmac": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-lineage", action="store_true")
    p.add_argument("--source-preflight", type=Path, required=True)
    p.add_argument("--source-preflight-sha256", required=True)
    p.add_argument("--audit", action="append", type=Path, required=True)
    p.add_argument("--audit-sha256", action="append", required=True)
    args = p.parse_args(argv)
    if not args.read_only_lineage:
        p.error("--read-only-lineage is required")
    if len(args.audit) != len(args.audit_sha256):
        p.error("every audit file requires a corresponding exact SHA")
    report = source_lineage_audit(
        args.source_preflight, args.source_preflight_sha256,
        list(zip(args.audit, args.audit_sha256, strict=True)))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

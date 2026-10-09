"""Fail-closed quarantine overlay for source-cluster readiness (no training).

Takes one pinned local lexical scan report and its exact source preflight.
It replays the quarantine *propagation* itself. A modified or misleading
candidate list, missing corpus scan, edited report or unmatched cohort
cannot silently release previously excluded source groups.

The scan is performed by the producer; its proof scope is limited to
within-approved-corpus lexical matching. It is NOT independently witnessed,
a semantic paraphrase proof or permission to train a classifier.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

from .k2_feature16_readiness import (
    hex_digest, load_json, private_file,
)
from .k2_cohort_preflight import SCHEMA as COHORT_SCHEMA

AUDIT_SCHEMA = "auto-finetune.dust-k2-cross-corpus-lexical-audit.v1"
MAX_AUX = 3


def bind_quarantine_overlay(
    manifest: dict, manifest_sha: str,
    audit_path: Path, expected_audit_sha: str,
) -> dict:
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError("invalid pinned cohort schema")
    if not hex_digest(manifest_sha) or not hex_digest(expected_audit_sha):
        raise ValueError("missing exact corpus/audit SHA256 pin")
    blob = private_file(audit_path, cap=128 * 1024)
    if hashlib.sha256(blob).hexdigest() != expected_audit_sha:
        raise ValueError("cross-corpus audit SHA256 mismatch")
    audit = load_json(blob)
    if not isinstance(audit, dict) or audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unknown cross-corpus audit schema")
    if (
        audit.get("mode") != "READ_ONLY_LEXICAL_SCREEN"
        or audit.get("source_preflight_sha256") != manifest_sha
        or audit.get("source_dataset_sha256")
            != manifest.get("source_dataset_sha256")
        or audit.get("source_model_config_sha256")
            != manifest.get("model_config_sha256")
        or audit.get("source_candidate_rows")
            != len(manifest.get("candidates", []))
        or audit.get("audit_provenance_scope")
            != "PRODUCER_LOCAL_LEXICAL_AUDIT_ONLY"
        or audit.get("classifier_training_authorized") is not False
        or audit.get("source_independence_certified") is not False
        or audit.get("semantic_paraphrase_screened") is not False
        or audit.get("all_corpora_exhaustively_audited") is not False
        or audit.get("receiver_key_custody_independent") is not False
        or audit.get("prompts_tokens_responses_in_output") is not False
    ):
        raise ValueError("audit provenance, scope or source hash inconsistent")
    groups = audit.get("aux_corpora")
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_AUX:
        raise ValueError("not enough approved auxiliary corpora were scanned")
    names = set()
    incomplete = False
    for row in groups:
        if not isinstance(row, dict):
            raise ValueError("malformed auxiliary scan result")
        name = row.get("filename")
        if (not isinstance(name, str) or not name or "/" in name
                or "\\" in name or name in names
                or not hex_digest(row.get("corpus_sha256"))):
            raise ValueError("invalid auxiliary corpus identity")
        names.add(name)
        count = row.get("pairs_screened")
        available = row.get("unique_pairs_reported")
        if (
            type(count) is not int or not 0 <= count <= 512
            or type(available) is not int or available < count
            or row.get("truncated") is not (available > count)
        ):
            raise ValueError("incorrect auxiliary truncation/accounting")
        incomplete |= available > count
    if audit.get("complete_auxiliary_coverage") is not (not incomplete):
        raise ValueError("auxiliary completeness status inconsistent")

    candidates = manifest["candidates"]
    by_index = {}
    for candidate in candidates:
        index = candidate.get("sample_index")
        cluster = candidate.get("near_duplicate_cluster_sha256")
        if (type(index) is not int or index < 0 or index in by_index
                or not hex_digest(cluster)):
            raise ValueError("invalid cohort index or near-duplicate identity")
        by_index[index] = candidate

    quarantined_indices = audit.get("quarantine_sample_indices")
    if (not isinstance(quarantined_indices, list)
            or any(type(x) is not int for x in quarantined_indices)
            or sorted(set(quarantined_indices)) != quarantined_indices
            or any(x not in by_index for x in quarantined_indices)):
        raise ValueError("cross-corpus quarantine indices invalid")
    matched_clusters = {
        by_index[index]["near_duplicate_cluster_sha256"]
        for index in quarantined_indices
    }
    expanded = sorted(
        index for index, row in by_index.items()
        if row["near_duplicate_cluster_sha256"] in matched_clusters
    )
    if expanded != quarantined_indices:
        raise ValueError("audit fails to quarantine whole related prompt cluster")
    expected_by_split = Counter(
        by_index[index]["group_split"] for index in quarantined_indices
    )
    if (
        audit.get("quarantine_near_duplicate_clusters")
            != len(matched_clusters)
        or audit.get("potentially_impacted_partition_counts")
            != dict(expected_by_split)
        or audit.get("source_cluster_contamination_review_required")
            is not bool(matched_clusters)
        or type(audit.get("cross_corpus_matched_source_indices")) is not int
        or type(audit.get("exactly_matched_source_indices")) is not int
        or not 0 <= audit["exactly_matched_source_indices"] <=
            audit["cross_corpus_matched_source_indices"] <= len(quarantined_indices)
    ):
        raise ValueError("inconsistent source contamination summary")
    return {
        "quarantined_clusters": matched_clusters,
        "quarantined_indices": quarantined_indices,
        "audited_auxiliary_corpus_count": len(groups),
        "truncated_auxiliary_corpora": sum(
            1 for g in groups if g["truncated"]),
        "lexical_scan_complete_for_supplied_corpora": not incomplete,
        "audit_sha256": expected_audit_sha,
        "independent_source_certification": False,
        "classifier_training_authorized": False,
    }

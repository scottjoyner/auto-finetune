"""Aggregate-only source-cluster readiness gate for K2 direction features.

Consumes private 16D records exported after causal PRE/POST replay. Requires
a pinned source-cohort preflight manifest. Enforces *source-cluster* separation
and real feature/label shape, but DOES NOT permit real-label classifier
training. All producer-only evidence remains HOLD until independently
isolated receiver-key custody, semantic source checks, and sufficient groups
are proven outside self-asserted JSON flags.

No prompts, tokens, directions, private HMAC identifiers, or per-row losses
are emitted in stdout. Never writes datasets, runs a model or fits a head.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat

from .k2_feature16_contract import FEATURE_NAMES, SCHEMA as FEATURES_SCHEMA
from .k2_cohort_preflight import SCHEMA as COHORT_SCHEMA
from .predictive_probe_contract import partition

EXPECTED_MINIMUM = {"train": 64, "validation": 16, "test": 32}
VALID_POPULATIONS = frozenset((8, 16, 32, 64))
EXACT_FIELDS = frozenset({
    "schema", "source_group_hmac_sha256", "model_revision_sha256",
    "source_partition_provisional", "direction_index",
    "features16_pre_probe", "beneficial_plus_direction",
    "true_plus_gain", "antithetic_slope", "timing_witness_scope",
    "receiver_key_isolated", "source_clusters_independently_verified",
    "classifier_training_authorized",
})
MAX_EPISODES = 256
MAX_FILE_BYTES = 1024 * 1024


def hex_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def private_file(path: Path, *, cap: int = MAX_FILE_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing/untrusted private research input")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077 or path.stat().st_size > cap:
        raise PermissionError("input requires private permissions and bounded size")
    return path.read_bytes()


def load_json(blob: bytes):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in research evidence")
            result[key] = value
        return result
    def no_constants(value):
        raise ValueError("nonfinite JSON value " + value)
    return json.loads(blob, object_pairs_hook=no_duplicates,
                      parse_constant=no_constants)


def validate_episode(path: Path, expected_model_sha: str):
    blob = private_file(path)
    lines = blob.splitlines()
    if not lines or len(lines) not in VALID_POPULATIONS:
        raise ValueError("incomplete or unsupported direction population")
    rows = [load_json(line) for line in lines]
    source_group = None
    split = None
    labels = []
    features = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != EXACT_FIELDS:
            raise ValueError("unexpected or unsafe 16D record fields")
        group = row["source_group_hmac_sha256"]
        if not hex_digest(group) or row["model_revision_sha256"] != expected_model_sha:
            raise ValueError("source HMAC or pretrained model revision mismatch")
        if source_group is None:
            source_group = group
            split = partition(group)
        if (group != source_group or row["source_partition_provisional"] != split
                or type(row["direction_index"]) is not int
                or row["direction_index"] != index):
            raise ValueError("mixed/replayed source or noncontiguous direction index")
        if (row["schema"] != FEATURES_SCHEMA
                or row["timing_witness_scope"] != "PRODUCER_CAUSAL_REPLAY_ONLY"
                or row["receiver_key_isolated"] is not False
                or row["source_clusters_independently_verified"] is not False
                or row["classifier_training_authorized"] is not False):
            raise ValueError("unexpected promotion or unsupported provenance")
        v = row["features16_pre_probe"]
        if (not isinstance(v, list) or len(v) != len(FEATURE_NAMES)
                or any(type(x) not in (int, float) or not math.isfinite(x)
                       or abs(x) > 1e6 for x in v)):
            raise ValueError("invalid or unsafe pre-probe feature vector")
        if v[0] != 0 or v[6] != 0.25 or v[7] != 0:
            raise ValueError("registered K2 v1 geometry invariants changed")
        if index < 4 and any(x != 0 for x in v[8:]):
            raise ValueError("first pre-probe batch leaked future history")
        if (type(row["beneficial_plus_direction"]) is not int
                or row["beneficial_plus_direction"] not in (0, 1)):
            raise ValueError("invalid binary direction label")
        gain = row["true_plus_gain"]
        slope = row["antithetic_slope"]
        if (type(gain) not in (int, float)
                or type(slope) not in (int, float)
                or not math.isfinite(gain) or not math.isfinite(slope)
                or abs(gain) > 100 or abs(slope) > 1000
                or row["beneficial_plus_direction"] != int(gain > 0)):
            raise ValueError("inconsistent or unsafe completed direction outcome")
        labels.append(row["beneficial_plus_direction"])
        features.append(v)
    for start in range(0, len(rows), 4):
        batch_histories = [tuple(v[8:]) for v in features[start:start + 4]]
        if len(set(batch_histories)) != 1:
            raise ValueError("current-batch outcomes entered PRE history")
    # Return only ephemeral source group for private preflight join; no
    # group IDs or private features escape the final aggregate result.
    return {
        "source_group": source_group, "split": split,
        "rows": len(rows), "positive": sum(labels),
        "features": features,
    }


def evaluate_readiness(feature_paths: list[Path], manifest_path: Path, *,
                       expected_manifest_sha: str,
                       expected_model_sha: str,
                       cross_corpus_audit_path: Path | None = None,
                       expected_cross_corpus_sha: str | None = None) -> dict:
    if (cross_corpus_audit_path is None) != (expected_cross_corpus_sha is None):
        raise ValueError("cross-corpus audit requires both file and digest")
    if not 1 <= len(feature_paths) <= MAX_EPISODES:
        raise ValueError("source episode count exceeds fixed acceptance bound")
    if not hex_digest(expected_manifest_sha) or not hex_digest(expected_model_sha):
        raise ValueError("unpinned cohort or model evidence")
    manifest_blob = private_file(manifest_path, cap=256 * 1024)
    if hashlib.sha256(manifest_blob).hexdigest() != expected_manifest_sha:
        raise ValueError("cohort SHA256 changed")
    manifest = load_json(manifest_blob)
    if manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError("source cohort schema mismatch")
    audit_overlay = None
    if cross_corpus_audit_path is not None:
        from .k2_crosscorpus_veto import bind_quarantine_overlay
        audit_overlay = bind_quarantine_overlay(
            manifest, expected_manifest_sha,
            cross_corpus_audit_path, expected_cross_corpus_sha)
    contaminated_clusters = (
        audit_overlay["quarantined_clusters"] if audit_overlay else set())
    candidates = manifest.get("candidates", [])
    if (not isinstance(candidates, list) or len(candidates) > 256):
        raise ValueError("invalid candidate cohort")
    allowed = {}
    eligible_clusters = {split: set() for split in EXPECTED_MINIMUM}
    cluster_splits = {}
    for candidate in candidates:
        if candidate.get("disposition") != "ELIGIBLE":
            continue
        group = candidate.get("episode_hmac_sha256")
        cluster = candidate.get("near_duplicate_cluster_sha256")
        if not hex_digest(group) or not hex_digest(cluster):
            raise ValueError("incomplete cohort HMAC cluster assignment")
        if cluster in contaminated_clusters:
            continue
        identity = (cluster, candidate.get("group_split"))
        if group in allowed and allowed[group] != identity:
            raise ValueError("one prompt group has incompatible cohort assignments")
        if identity[1] not in EXPECTED_MINIMUM:
            raise ValueError("invalid partition name")
        if cluster in cluster_splits and cluster_splits[cluster] != identity[1]:
            raise ValueError("near-duplicate cluster crosses pinned partitions")
        cluster_splits[cluster] = identity[1]
        eligible_clusters[identity[1]].add(cluster)
        allowed[group] = identity
    observed_groups = set()
    observed_clusters = set()
    counts = Counter()
    positive = Counter()
    records = 0
    observations = []
    for path in feature_paths:
        episode = validate_episode(path, expected_model_sha)
        group = episode["source_group"]
        if group not in allowed:
            raise ValueError("source absent from eligible pinned cohort or quarantined by cross-corpus audit")
        cluster, cohort_split = allowed[group]
        if episode["split"] != cohort_split:
            raise ValueError("provisional partition differs from pinned cohort")
        if group in observed_groups or cluster in observed_clusters:
            raise ValueError("source or near-duplicate cluster reused across episodes")
        observed_groups.add(group)
        observed_clusters.add(cluster)
        counts[cohort_split] += 1
        positive[cohort_split] += episode["positive"]
        records += episode["rows"]
        observations.extend(episode["features"])
    # Variance is computed across rows only as an observational diagnostic,
    # not as evidence of statistical independence or heldout generalization.
    varying_columns = sum(
        len({row[i] for row in observations}) > 1
        for i in range(len(FEATURE_NAMES))
    )
    minimum = {
        split: counts[split] >= required
        for split, required in EXPECTED_MINIMUM.items()
    }
    return {
        "schema": "auto-finetune.dust-k2-feature16-readiness.v1",
        "status": "HOLD_NO_INDEPENDENT_CUSTODY_OR_SUFFICIENT_SOURCE_GROUPS",
        "mode": "READ_ONLY_PROVISIONAL_SOURCE_CLUSTER_AUDIT",
        "independent_source_clusters_observed": len(observed_clusters),
        "complete_direction_rows": records,
        "source_cluster_counts": dict(sorted(
            (split, counts[split]) for split in EXPECTED_MINIMUM)),
        "positive_direction_labels": dict(sorted(
            (split, positive[split]) for split in EXPECTED_MINIMUM)),
        "independent_source_minimum": EXPECTED_MINIMUM,
        "available_eligible_source_clusters_in_pinned_preflight": {
            split: len(eligible_clusters[split])
            for split in EXPECTED_MINIMUM
        },
        "preflight_source_cluster_deficit_to_minimum": {
            split: max(0, EXPECTED_MINIMUM[split] - len(eligible_clusters[split]))
            for split in EXPECTED_MINIMUM
        },
        "collected_source_cluster_deficit_to_minimum": {
            split: max(0, EXPECTED_MINIMUM[split] - counts[split])
            for split in EXPECTED_MINIMUM
        },
        "preflight_adequacy_for_training_design": all(
            len(eligible_clusters[split]) >= required
            for split, required in EXPECTED_MINIMUM.items()
        ),
        "minimum_source_counts_satisfied": minimum,
        "all_source_count_minima_met": all(minimum.values()),
        "varying_feature_columns_observed": varying_columns,
        "feature_columns_total": len(FEATURE_NAMES),
        "source_lexical_cohort_preflight_only": True,
        "cross_corpus_audit_supplied": audit_overlay is not None,
        "cross_corpus_lexical_audit_coverage": (
            "NOT_RUN" if audit_overlay is None else
            "INCOMPLETE_TRUNCATED" if not audit_overlay[
                "lexical_scan_complete_for_supplied_corpora"] else
            "COMPLETE_FOR_SUPPLIED_CORPORA_ONLY"),
        "cross_corpus_audited_auxiliary_corpora": (
            audit_overlay["audited_auxiliary_corpus_count"] if audit_overlay else 0),
        "cross_corpus_quarantined_source_clusters": len(contaminated_clusters),
        "independent_audit_of_cross_corpus_matching": False,
        "cross_corpus_semantic_contamination_verified": False,
        "receiver_key_isolation_verified": False,
        "independent_forward_loss_numeric_witness": False,
        "real_label_classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
        "raw_prompt_or_hmac_in_summary": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-readiness", action="store_true")
    p.add_argument("--features", type=Path, action="append", required=True)
    p.add_argument("--source-preflight", type=Path, required=True)
    p.add_argument("--source-preflight-sha256", required=True)
    p.add_argument("--expected-model-sha256", required=True)
    p.add_argument("--cross-corpus-audit", type=Path,
                   help="optional SHA-pinned local lexical scan quarantine")
    p.add_argument("--cross-corpus-audit-sha256",
                   help="required when supplying a cross-corpus audit")
    args = p.parse_args(argv)
    if not args.read_only_readiness:
        p.error("explicit --read-only-readiness required")
    result = evaluate_readiness(
        args.features, args.source_preflight,
        expected_manifest_sha=args.source_preflight_sha256,
        expected_model_sha=args.expected_model_sha256,
        cross_corpus_audit_path=args.cross_corpus_audit,
        expected_cross_corpus_sha=args.cross_corpus_audit_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

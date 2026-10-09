"""Produce a versioned, restricted 16D K2 direction research dataset.

This is a validation and export tool, NOT a classifier trainer. It refuses
to export unless an independent causal PRE/POST replay passes. Eight candidate
geometry proxies and eight *previous-completed-probe-only* history features
are carried without training any model or selecting perturbations.

Feature order is immutable v1: geometry (8) then causal history (8).
The source-prompt HMAC is an opaque group identifier, NOT an independently
verified near-duplicate/source split. Exported files are confidential and
must remain on a private local SSD. No raw prompts, tokens, directions or
activations are exported. NEVER put records in GitHub Actions artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from .k2_causal_evidence_verify import verify
from .k2_causal_feature_history import FEATURES as HISTORY_NAMES
from .predictive_probe_contract import validate_record, partition

SCHEMA = "auto-finetune.dust-k2-real-direction-features16.v1"
GEOMETRY_NAMES = (
    "geometry_unavailable_momentum_zero",
    "direction_anchor_projection",
    "direction_prior_projection",
    "direction_activation_spread_quadratic",
    "sigma_scaled_activation_spread",
    "direction_spread_absolute_mean",
    "sigma",
    "geometry_unavailable_history_zero",
)
FEATURE_NAMES = GEOMETRY_NAMES + tuple("history_" + name for name in HISTORY_NAMES)
assert len(FEATURE_NAMES) == 16
MAX_ROWS = 64
MAX_BYTES = 1024 * 1024


def extract_verified_episode(events_path: Path, derived_path: Path,
                             summary_path: Path) -> tuple[list[dict], dict]:
    """Replay-first: only finished, full-K, SHA-correct episodes are exported.

    PRE historical values were previously recomputed by verify(), so these
    feature rows do not trust a producer-only 'features_are_early' boolean.
    """
    audit = verify(events_path, derived_path, summary_path)
    if (
        audit["result"] != "PASS_PRODUCER_CAUSAL_REPLAY_ONLY"
        or audit["history_pre_only_recomputed"] is not True
        or audit["original_label_schema_preserved"] is not True
        or audit["receiver_signing_key_isolation_accepted"] is not False
    ):
        raise ValueError("causal replay did not pass correct scope")
    events = [json.loads(x) for x in events_path.read_text().splitlines()]
    labels = [json.loads(x) for x in derived_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    if len(labels) != audit["observed_directions"] or len(labels) > MAX_ROWS:
        raise ValueError("incomplete or oversized direction population")
    expected_group = summary["source_episode_hmac_sha256"]
    expected_model = summary["model_revision_sha256"]
    pre = {e["candidate_index"]: e for e in events if e["phase"] == "PRE"}
    if set(pre) != set(range(len(labels))):
        raise ValueError("missing or duplicated PRE indices")
    source_split = partition(expected_group)
    records = []
    seen = set()
    for index, label in enumerate(labels):
        validated = validate_record(label)
        p = pre[index]
        if (
            validated["candidate"] != index
            or validated["episode"] != expected_group
            or validated["model_revision_sha256"] != expected_model
            or validated["split"] != source_split
            or validated["label"] != int(label["loss_plus"] < label["loss_clean"])
        ):
            raise ValueError("unexpected source or label")
        # Check immutable, fully numeric and bounded 16D record.
        geometry = p["features_pre_probe"]
        history = p["history_features_pre_probe"]
        if len(geometry) != 8 or len(history) != 8:
            raise ValueError("unexpected feature dimensionality")
        features = tuple(float(x) for x in geometry + history)
        if not all(math.isfinite(x) and abs(x) <= 1e6 for x in features):
            raise ValueError("nonfinite feature or out of bounds")
        # v1 geometry contains two unavailable historical placeholders.
        if features[0] != 0.0 or features[7] != 0.0:
            raise ValueError("unregistered geometry feature-v1 meaning changed")
        if index < 4 and any(x != 0.0 for x in features[8:]):
            raise ValueError("first batch must have no prior history")
        if index in seen:
            raise ValueError("duplicate direction index")
        seen.add(index)
        records.append({
            "schema": SCHEMA,
            "source_group_hmac_sha256": expected_group,
            "model_revision_sha256": expected_model,
            "source_partition_provisional": source_split,
            "direction_index": index,
            "features16_pre_probe": list(features),
            "beneficial_plus_direction": validated["label"],
            "true_plus_gain": validated["gain"],
            "antithetic_slope": validated["antithetic_slope"],
            "timing_witness_scope": "PRODUCER_CAUSAL_REPLAY_ONLY",
            "receiver_key_isolated": False,
            "source_clusters_independently_verified": False,
            "classifier_training_authorized": False,
        })
    return records, {
        **audit,
        "feature_schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "direction_rows": len(records),
        "source_episode_count": 1,
        "source_partition_provisional": source_split,
        "history8_post_only": True,
        "candidate_geometry8_present": True,
        "legacy_zero_placeholder_count": 2,
        "semantic_near_duplicates_verified": False,
        "training_data_authorized": False,
        "classifier_training_authorized": False,
        "production_authorized": False,
    }


def export_private(events: Path, derived: Path, summary: Path,
                   destination: Path) -> dict:
    if (destination.exists() or destination.is_symlink()
            or not destination.parent.is_dir()
            or destination.parent.stat().st_mode & 0o077):
        raise PermissionError("export requires new file in mode-700 directory")
    records, manifest = extract_verified_episode(events, derived, summary)
    os.umask(0o077)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        for row in records:
            out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    manifest["private_export_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest["private_export_mode"] = "0600"
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate-and-export-only", action="store_true")
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--derived", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.validate_and_export_only:
        p.error("must explicitly choose --validate-and-export-only")
    report = export_private(
        args.events, args.derived, args.summary, args.private_output)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

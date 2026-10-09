"""Immutable independent human adjudication gate — fail closed by default.

Accepts two distinct human review receipts, each SHA-bound to the exact
queue and with independently supplied judgments on queue pair_id only.
No prompts, model scores, source HMACs, raw labels or reviewers' private
identity data enter the aggregate result. Disagreements remain unresolved.
Even 100% agreement does NOT authorize classifier training, dataset rights,
receiver signing-key trust or automated train/eval split changes.

The actual human-review process occurs locally under separate human
credentials. No model generated/self-asserted annotation is sufficient.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .k2_private_semantic_review_queue import (
    SCHEMA as QUEUE_SCHEMA, validate_private_json)

SCHEMA = "auto-finetune.dust-k2-human-semantic-adjudication.v1"
LABELS = frozenset(("SAME_INTENT", "DIFFERENT_INTENT", "UNCERTAIN"))
MAX_QUEUE = 500


def validate_review_receipt(document, queue_sha, ids):
    if (not isinstance(document, dict)
            or document.get("schema") !=
                "auto-finetune.dust-k2-independent-human-review.v1"
            or document.get("queue_sha256") != queue_sha
            or document.get("model_generated_labels") is not False
            or document.get("reviewer_attests_human_review") is not True):
        raise ValueError("review was not independently attested to this exact queue")
    reviewer = document.get("reviewer_id")
    if not isinstance(reviewer, str) or not 3 <= len(reviewer) <= 64:
        raise ValueError("missing distinct reviewer identity")
    decisions = document.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != len(ids):
        raise ValueError("review lacks complete pair coverage")
    mapping = {}
    for item in decisions:
        if (not isinstance(item, dict)
                or set(item) != {"pair_id_sha256", "decision"}
                or item.get("decision") not in LABELS):
            raise ValueError("invalid review decision row")
        i = item.get("pair_id_sha256")
        if i not in ids or i in mapping:
            raise ValueError("unexpected or duplicate review pair")
        mapping[i] = item["decision"]
    if set(mapping) != ids:
        raise ValueError("omitted queue review pair")
    return reviewer, mapping


def adjudicate_private_queue(queue_path: Path, queue_sha: str,
                              reviewer_a: Path, reviewer_a_sha: str,
                              reviewer_b: Path, reviewer_b_sha: str):
    queue = validate_private_json(queue_path, queue_sha)
    if (queue.get("schema") != QUEUE_SCHEMA
            or queue.get("prior_aggregate_reconciled") is not True
            or queue.get("human_decisions_observed") != 0
            or queue.get("classifier_training_authorized") is not False
            or queue.get("reviewer_identified_or_adjudicated") is not False):
        raise ValueError("untrusted or already-promoted original review queue")
    items = queue.get("review_candidates")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_QUEUE:
        raise ValueError("review queue not populated")
    ids = {x["pair_id_sha256"] for x in items}
    if len(ids) != len(items):
        raise ValueError("duplicate pair IDs in private review queue")
    a = validate_private_json(reviewer_a, reviewer_a_sha)
    b = validate_private_json(reviewer_b, reviewer_b_sha)
    aid, decisions_a = validate_review_receipt(a, queue_sha, ids)
    bid, decisions_b = validate_review_receipt(b, queue_sha, ids)
    if aid == bid:
        raise ValueError("same human cannot sign both independent reviews")
    outcomes = Counter()
    for identity in ids:
        first, second = decisions_a[identity], decisions_b[identity]
        if first == second == "SAME_INTENT":
            outcomes["AGREED_SAME_INTENT"] += 1
        elif first == second == "DIFFERENT_INTENT":
            outcomes["AGREED_DIFFERENT_INTENT"] += 1
        else:
            outcomes["UNRESOLVED_OR_UNCERTAIN"] += 1
    return {
        "schema": SCHEMA,
        "source_queue_sha256": queue_sha,
        "reviewer_receipts_shas": sorted((reviewer_a_sha, reviewer_b_sha)),
        "distinct_attested_reviewer_count": 2,
        "pairs_reviewed_by_both": len(ids),
        "agreed_same_intent_candidates": outcomes["AGREED_SAME_INTENT"],
        "agreed_different_intent_candidates": outcomes["AGREED_DIFFERENT_INTENT"],
        "unresolved_or_uncertain_candidates":
            outcomes["UNRESOLVED_OR_UNCERTAIN"],
        "two_human_attestations_not_cryptographically_independent": True,
        "human_interrater_review_is_not_semantic_detector_recall": True,
        "automatic_source_reassignment": False,
        "semantic_dataset_independence_certified": False,
        "data_reuse_rights_approved": False,
        "receiver_key_custody_independent": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
        "no_private_pair_data_or_reviewer_id_in_aggregate": True,
    }

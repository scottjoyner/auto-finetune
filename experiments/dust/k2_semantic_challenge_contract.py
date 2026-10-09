"""Frozen CPU-only paraphrase challenge and semantic-audit decision contract.

The preset challenges are synthetic instructions without user data. They
measure this specific encoder's ability to catch known paraphrases and to
avoid tempting hard negatives. Neither threshold nor benchmark is ground
truth for the user's private corpus. Failing the challenge leaves the
semantic screening gate HOLD, never suppresses dangerous candidates.

All real prompt identifiers are keyed opaque HMACs in private SSD receipts.
"""
from __future__ import annotations

from collections import Counter
import math

SCHEMA = "auto-finetune.dust-k2-semantic-challenge.v1"
THRESHOLDS = (0.75, 0.85, 0.92)
PAIRS = (
    ("How do I reset a forgotten account password?",
     "I cannot remember my login password; how can I regain access?", 1),
    ("Write Python code that sorts a list of strings.",
     "Produce a Python snippet to alphabetize several text values.", 1),
    ("Explain how to turn a JSON file into CSV format.",
     "What steps convert JSON records to comma-separated values?", 1),
    ("How can I reduce the memory usage of an inference process?",
     "What are ways to lower RAM consumption when running a model?", 1),
    ("How should I back up a database before migrating it?",
     "What is a safe way to make a database copy ahead of a move?", 1),
    ("How do I measure a classifier's precision and recall?",
     "What metrics quantify positive prediction correctness and coverage?", 1),
    ("Describe how to prevent data leakage between train and test.",
     "How can similar examples be kept separate across model splits?", 1),
    ("What is the procedure to restart a failed Linux service?",
     "How can I bring a stopped systemd daemon back online?", 1),
    ("How do I verify a download's SHA-256 checksum?",
     "How can I check that a file matches its expected hash digest?", 1),
    ("Explain the tradeoffs of smaller quantized neural network weights.",
     "What changes when model parameters are stored at lower precision?", 1),
    ("How do I reset a forgotten account password?",
     "How do I create a strong new password before registration?", 0),
    ("Write Python code that sorts a list of strings.",
     "Write Python code that reverses a linked list.", 0),
    ("Explain how to turn a JSON file into CSV format.",
     "Explain how to sign a JSON Web Token for authentication.", 0),
    ("How can I reduce the memory usage of an inference process?",
     "How can I lower CPU clock speed to conserve battery power?", 0),
    ("How should I back up a database before migrating it?",
     "How should I delete a database after confirming migration?", 0),
    ("How do I measure a classifier's precision and recall?",
     "How do I calculate regression mean squared error?", 0),
    ("Describe how to prevent data leakage between train and test.",
     "How do I increase the random seed variation across experiments?", 0),
    ("What is the procedure to restart a failed Linux service?",
     "How can I configure a Linux service to launch only at boot?", 0),
    ("How do I verify a download's SHA-256 checksum?",
     "How do I encrypt a file using AES-256?", 0),
    ("Explain the tradeoffs of smaller quantized neural network weights.",
     "Explain how to compress photographs as JPEG images.", 0),
)


def evaluate_challenge(scores: list[float]) -> dict:
    if len(scores) != len(PAIRS) or any(
        type(x) not in (int, float) or not math.isfinite(x)
        or not -1.00001 <= x <= 1.00001 for x in scores
    ):
        raise ValueError("expected 20 bounded finite challenge cosines")
    results = {}
    for threshold in THRESHOLDS:
        tp = fp = tn = fn = 0
        for score, (_, _, truth) in zip(scores, PAIRS, strict=True):
            predicted = score >= threshold
            if truth and predicted: tp += 1
            elif truth and not predicted: fn += 1
            elif not truth and predicted: fp += 1
            else: tn += 1
        recall = tp / (tp + fn)
        false_positive_rate = fp / (fp + tn)
        results[str(threshold)] = {
            "threshold": threshold,
            "true_positive": tp, "false_negative": fn,
            "false_positive": fp, "true_negative": tn,
            "recall": recall,
            "false_positive_rate": false_positive_rate,
            "challenge_pass_for_screening_research_only":
                recall >= 0.7 and false_positive_rate <= 0.2,
        }
    return {
        "schema": SCHEMA,
        "positive_challenge_pairs": 10,
        "hard_negative_pairs": 10,
        "prespecified_threshold_results": results,
        "labels_are_synthetic_not_real_corpus_ground_truth": True,
        "semantic_independence_certified": False,
        "classifier_training_authorized": False,
    }


def summarize_collisions(
    *, source_groups: list[str], auxiliary_groups: list[str],
    auxiliary_splits: list[str], cross_hits: list[tuple[int, int, float]],
    within_hits: list[tuple[int, int, float]], threshold: float
) -> dict:
    """Aggregate only; never emit prompt strings or source HMACs."""
    if threshold not in THRESHOLDS:
        raise ValueError("unregistered semantic triage threshold")
    if len(auxiliary_groups) != len(auxiliary_splits):
        raise ValueError("auxiliary families and split labels mismatch")
    if any(x not in ("train", "validation", "test")
           for x in auxiliary_splits):
        raise ValueError("unrecognized candidate partition")
    source = set()
    aux_cross = set()
    for i, j, similarity in cross_hits:
        if (not 0 <= i < len(source_groups)
                or not 0 <= j < len(auxiliary_groups)
                or not math.isfinite(similarity)
                or similarity < threshold):
            raise ValueError("invalid cross-corpus candidate similarity")
        source.add(i)
        aux_cross.add(j)
    different_families = 0
    cross_split_edges = 0
    implicated = set()
    already_clustered_edges = 0
    for i, j, similarity in within_hits:
        if (not 0 <= i < j < len(auxiliary_groups)
                or not math.isfinite(similarity)
                or similarity < threshold):
            raise ValueError("invalid within-source candidate similarity")
        if auxiliary_groups[i] == auxiliary_groups[j]:
            already_clustered_edges += 1
            continue
        different_families += 1
        implicated.add(auxiliary_groups[i])
        implicated.add(auxiliary_groups[j])
        cross_split_edges += (auxiliary_splits[i] != auxiliary_splits[j])
    return {
        "threshold": threshold,
        "original_source_prompts_with_embedding_neighbour": len(source),
        "auxiliary_prompts_near_original_source": len(aux_cross),
        "cross_corpus_embedding_edges": len(cross_hits),
        "within_auxiliary_existing_family_edges": already_clustered_edges,
        "within_auxiliary_new_cross_family_edges": different_families,
        "within_auxiliary_new_cross_split_edges": cross_split_edges,
        "lexical_candidate_families_implicated": len(implicated),
        "all_embedding_neighbours_require_review": True,
        "semantic_equivalence_certified": False,
        "heldout_splits_authorized": False,
        "classifier_training_authorized": False,
    }

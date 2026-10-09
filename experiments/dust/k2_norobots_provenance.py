"""SHA-bound metadata-only provenance triage for local No Robots candidates.

The local 'norobots/<category>' source tags and ~9500 conversation rows
corroborate, but do not cryptographically establish, descent from the
HuggingFaceH4/no_robots train split. Its upstream dataset card states
CC BY-NC 4.0; this module NEVER converts that evidence to permission
for commercial training or source-holdout certification.

No messages or user prompts are read into the report, stdout or GitHub.
This performs no model inference and no classifier training.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

SCHEMA = "auto-finetune.dust-k2-no-robots-provenance.v1"
KNOWN_CATEGORIES = frozenset((
    "Summarize", "Generation", "Rewrite", "Open QA", "Closed QA",
    "Chat", "Brainstorm", "Coding", "Classify", "Extract"))
SOURCE_CARD = "https://huggingface.co/datasets/HuggingFaceH4/no_robots"
UPSTREAM_LICENSE = "CC-BY-NC-4.0"


def summarize_provenance(path: Path, *, source_sha256: str):
    if (not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("source must be pinned to a lowercase SHA256")
    if path.is_symlink() or not path.is_file():
        raise ValueError("unsafe or missing corpus path")
    digest = hashlib.sha256()
    categories = Counter()
    rows = 0
    unexpected = 0
    declared_rights = Counter()
    with path.open("rb") as fd:
        for raw in fd:
            digest.update(raw)
            rows += 1
            if rows > 25000 or len(raw) > 128 * 1024:
                raise ValueError("unbounded corpus provenance record")
            record = json.loads(raw)
            if not isinstance(record, dict) or not isinstance(
                    record.get("messages"), list):
                raise ValueError("corpus row not in supported message schema")
            source = record.get("source")
            if (not isinstance(source, str)
                    or not source.startswith("norobots/")):
                unexpected += 1
                continue
            category = source.removeprefix("norobots/")
            if category in KNOWN_CATEGORIES:
                categories[category] += 1
            else:
                unexpected += 1
            # A local license/source field is not proof of an upstream grant.
            for name in ("license", "dataset_license", "rights"):
                if name in record:
                    declared_rights["local_unverified_field"] += 1
    if digest.hexdigest() != source_sha256:
        raise ValueError("pinned local corpus SHA256 mismatch")
    return {
        "schema": SCHEMA,
        "mode": "READ_ONLY_METADATA_PROVENANCE",
        "local_source_sha256": source_sha256,
        "local_conversation_rows": rows,
        "recognized_norobots_labeled_rows": sum(categories.values()),
        "unexpected_source_category_rows": unexpected,
        "category_label_counts": dict(sorted(categories.items())),
        "local_rights_field_rows": declared_rights["local_unverified_field"],
        "upstream_similarity_basis":
            "norobots category tags and 9500 conversations; exact lineage not proven",
        "candidate_upstream_dataset": "HuggingFaceH4/no_robots",
        "candidate_upstream_card": SOURCE_CARD,
        "candidate_upstream_license": UPSTREAM_LICENSE,
        "upstream_dataset_license_scope":
            "ATTRIBUTION_AND_NONCOMMERCIAL_RESTRICTIONS_APPLY",
        "local_corpus_transformation_provenance_verified": False,
        "local_corpus_license_grant_verified": False,
        "commercial_training_rights_approved": False,
        "noncommercial_research_subject_to_license_review_only": True,
        "semantic_source_independence_verified": False,
        "new_heldout_splits_approved": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
        "prompt_response_or_token_content_in_report": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-only-provenance", action="store_true")
    p.add_argument("--local-corpus", required=True, type=Path)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--private-output", required=True, type=Path)
    args = p.parse_args(argv)
    if not args.read_only_provenance:
        p.error("explicit --read-only-provenance flag required")
    destination = args.private_output
    if (destination.exists() or destination.is_symlink()
            or not destination.parent.is_dir()
            or destination.parent.stat().st_mode & 0o077):
        p.error("new file in existing private directory required")
    report = summarize_provenance(
        args.local_corpus, source_sha256=args.expected_source_sha256)
    os.umask(0o077)
    with destination.open("x", encoding="utf-8") as fd:
        fd.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

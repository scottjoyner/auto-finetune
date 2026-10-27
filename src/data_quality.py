"""Real-time data quality monitoring for the pipeline.

Scores incoming data on multiple dimensions:
  - Completeness: required fields present
  - Validity: values in expected ranges
  - Uniqueness: no duplicate sessions
  - Consistency: cross-field coherence
  - Freshness: data recency

Provides quality gates that can block bad data from entering training.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class QualityScore:
    """Quality assessment for a single data record."""
    session_id: str
    completeness: float = 0.0  # 0-1
    validity: float = 0.0
    uniqueness: float = 0.0
    consistency: float = 0.0
    freshness: float = 0.0
    overall: float = 0.0
    issues: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class QualityReport:
    """Aggregate quality report for a batch of records."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    min_score: float = 0.0
    issues_summary: dict[str, int] = field(default_factory=dict)
    timestamp: float = 0.0


REQUIRED_FIELDS = {"messages"}
MIN_MESSAGE_COUNT = 2
MAX_MESSAGE_COUNT = 1000


def score_session(session: dict) -> QualityScore:
    """Score a single session record on all quality dimensions."""
    score = QualityScore(session_id=session.get("session_id", "unknown"))
    issues = []

    # Check messages have content
    msgs = session.get("messages", [])
    if not msgs:
        issues.append("no_messages")

    # Detect format: raw (parts) vs formatted (content)
    is_formatted = any("content" in m for m in msgs if isinstance(m, dict)) and not any("parts" in m for m in msgs if isinstance(m, dict))

    # --- Completeness ---
    present = sum(1 for f in REQUIRED_FIELDS if f in session and session[f] is not None)
    if is_formatted:
        score.completeness = 1.0 if msgs else 0.0
    else:
        score.completeness = present / len(REQUIRED_FIELDS)
    if score.completeness < 1.0 and not is_formatted:
        missing = REQUIRED_FIELDS - set(session.keys())
        issues.append(f"missing_fields:{','.join(missing)}")
    if not msgs:
        score.completeness = 0.0

    # --- Validity ---
    validity_checks = 0
    validity_total = 0

    # Message count in range
    validity_total += 1
    if MIN_MESSAGE_COUNT <= len(msgs) <= MAX_MESSAGE_COUNT:
        validity_checks += 1
    else:
        issues.append(f"message_count_out_of_range:{len(msgs)}")

    # Each message has role and parts
    for m in msgs:
        validity_total += 1
        if isinstance(m, dict) and "role" in m:
            if is_formatted or "parts" in m:
                validity_checks += 1
            else:
                issues.append("malformed_message")
                break
        else:
            issues.append("malformed_message")
            break

    # Parts are non-empty
    for m in msgs:
        validity_total += 1
        if is_formatted:
            content = m.get("content", "") if isinstance(m, dict) else ""
            if content and isinstance(content, str) and len(content) > 0:
                validity_checks += 1
        else:
            parts = m.get("parts", []) if isinstance(m, dict) else []
            if parts and any((isinstance(p, dict) and (p.get("text") or p.get("type") == "tool")) for p in parts):
                validity_checks += 1

    score.validity = validity_checks / max(1, validity_total)

    # --- Uniqueness ---
    # Hash the conversation content
    content_hash = hashlib.sha256(
        json.dumps(session.get("messages", []), sort_keys=True).encode()
    ).hexdigest()
    score.uniqueness = 1.0  # Will be checked against a seen set externally

    # --- Consistency ---
    consistency_checks = 0
    consistency_total = 0

    # Roles alternate (user/assistant)
    consistency_total += 1
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if len(roles) >= 2:
        # Check that not all roles are the same
        if len(set(roles)) > 1:
            consistency_checks += 1
        else:
            issues.append("single_role_conversation")

    # Source is valid
    consistency_total += 1
    # For formatted datasets, source may not be present
    if "source" in session:
        valid_sources = {"opencode", "hermes", "neo4j-traces", "combined"}
        if session.get("source") in valid_sources:
            consistency_checks += 1
        else:
            issues.append(f"unknown_source:{session.get('source')}")
    else:
        # Formatted dataset - skip source check
        consistency_checks += 1

    score.consistency = consistency_checks / max(1, consistency_total)

    # --- Freshness ---
    # Check if time_created is recent (within last 90 days)
    tc = session.get("time_created", 0)
    if isinstance(tc, (int, float)) and tc > 0:
        age_days = (time.time() - tc) / 86400
        if age_days < 90:
            score.freshness = 1.0
        elif age_days < 365:
            score.freshness = 0.7
        else:
            score.freshness = 0.3
    else:
        score.freshness = 0.5  # Unknown freshness

    # --- Overall ---
    score.overall = (
        score.completeness * 0.3 +
        score.validity * 0.3 +
        score.consistency * 0.2 +
        score.freshness * 0.1 +
        score.uniqueness * 0.1
    )

    score.issues = issues
    score.passed = score.overall >= 0.6 and score.completeness >= 0.66

    return score


def assess_batch(sessions: list[dict]) -> QualityReport:
    """Assess a batch of sessions and return an aggregate report."""
    report = QualityReport(total=len(sessions), timestamp=time.time())
    scores = []
    seen_hashes = set()
    issue_counts: dict[str, int] = {}

    for session in sessions:
        score = score_session(session)

        # Check uniqueness against seen set
        content_hash = hashlib.sha256(
            json.dumps(session.get("messages", []), sort_keys=True).encode()
        ).hexdigest()
        if content_hash in seen_hashes:
            score.uniqueness = 0.0
            score.issues.append("duplicate")
            score.overall = score.overall * 0.9  # Penalize duplicates
            if score.overall < 0.6:
                score.passed = False
        seen_hashes.add(content_hash)

        scores.append(score)
        if score.passed:
            report.passed += 1
        else:
            report.failed += 1

        for issue in score.issues:
            key = issue.split(":")[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1

    if scores:
        report.avg_score = sum(s.overall for s in scores) / len(scores)
        report.min_score = min(s.overall for s in scores)

    report.issues_summary = issue_counts
    return report


def filter_quality(sessions: list[dict], min_score: float = 0.6) -> list[dict]:
    """Filter sessions by quality score, returning only those that pass."""
    result = []
    for session in sessions:
        score = score_session(session)
        if score.overall >= min_score and score.passed:
            result.append(session)
    return result


def quality_gate(cfg: Config, input_path: str, output_path: str,
                 min_score: float = 0.6) -> dict:
    """Apply quality gate to a dataset file.

    Reads sessions from input_path, scores them, writes passing ones to output_path.
    Returns a summary dict.
    """
    sessions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    report = assess_batch(sessions)
    passing = filter_quality(sessions, min_score)

    # Write passing sessions
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for s in passing:
            f.write(json.dumps(s) + "\n")

    return {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "avg_score": round(report.avg_score, 3),
        "min_score": round(report.min_score, 3),
        "issues": report.issues_summary,
        "output": output_path,
    }


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Data quality assessment")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output filtered JSONL file")
    parser.add_argument("--min-score", type=float, default=0.6, help="Minimum quality score")
    parser.add_argument("--report", action="store_true", help="Print detailed report")
    args = parser.parse_args(argv)

    sessions = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sessions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    report = assess_batch(sessions)
    print(f"[quality] Total: {report.total}, Passed: {report.passed}, "
          f"Failed: {report.failed}")
    print(f"[quality] Avg score: {report.avg_score:.3f}, Min: {report.min_score:.3f}")
    if report.issues_summary:
        print(f"[quality] Issues: {report.issues_summary}")

    if args.output:
        passing = filter_quality(sessions, args.min_score)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            for s in passing:
                f.write(json.dumps(s) + "\n")
        print(f"[quality] Wrote {len(passing)} passing sessions to {args.output}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

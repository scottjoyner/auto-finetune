"""Feedback loop: use inference results to identify bad training examples.

Collects inference feedback (success/failure, user corrections, error signals)
and uses them to:
  1. Flag low-quality training examples
  2. Generate contrastive repair pairs
  3. Update quality scores
  4. Trigger retraining on problematic patterns
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class FeedbackEntry:
    """A single feedback entry from inference."""
    entry_id: str
    session_id: str
    model_version: str
    input_text: str
    output_text: str
    expected_text: str | None = None
    score: float = 0.0  # 0-1, higher is better
    feedback_type: str = "auto"  # auto, user, error, correction
    issues: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackSummary:
    """Summary of feedback for a training dataset."""
    total_entries: int = 0
    avg_score: float = 0.0
    low_quality_count: int = 0
    issue_distribution: dict[str, int] = field(default_factory=dict)
    flagged_sessions: list[str] = field(default_factory=list)
    repair_pairs: int = 0


class FeedbackCollector:
    """Collects and analyzes inference feedback."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.feedback_path = os.path.join(cfg.path("analysis_dir"), "feedback.jsonl")
        self.summary_path = os.path.join(cfg.path("analysis_dir"), "feedback-summary.json")
        self.entries: list[FeedbackEntry] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.feedback_path):
            try:
                with open(self.feedback_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.entries.append(FeedbackEntry(**data))
            except Exception:
                pass

    def _save_entry(self, entry: FeedbackEntry) -> None:
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        with open(self.feedback_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def record(self, session_id: str, model_version: str, input_text: str,
               output_text: str, score: float = 0.0,
               feedback_type: str = "auto",
               expected_text: str | None = None,
               issues: list[str] | None = None,
               metadata: dict | None = None) -> str:
        """Record a feedback entry."""
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("feedback score must be finite and between 0 and 1")
        entry_id = f"fb-{int(time.time() * 1000)}"
        entry = FeedbackEntry(
            entry_id=entry_id,
            session_id=session_id,
            model_version=model_version,
            input_text=input_text[:2000],
            output_text=output_text[:2000],
            expected_text=expected_text[:2000] if expected_text else None,
            score=score,
            feedback_type=feedback_type,
            issues=issues or [],
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self.entries.append(entry)
        self._save_entry(entry)
        return entry_id

    def _detect_issues(self, entry: FeedbackEntry) -> list[str]:
        """Auto-detect issues in an inference result."""
        issues = list(entry.issues)
        if entry.score < 0.3:
            issues.append("low_score")
        if entry.output_text and len(entry.output_text) < 10:
            issues.append("too_short")
        if entry.output_text and len(entry.output_text) > 5000:
            issues.append("too_long")
        # Check for repetition
        if entry.output_text:
            words = entry.output_text.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    issues.append("repetitive")
        # Check for error patterns
        error_patterns = ["error", "exception", "traceback", "failed", "cannot"]
        lower = entry.output_text.lower()
        if any(p in lower for p in error_patterns):
            issues.append("error_in_output")
        return list(dict.fromkeys(issues))

    def analyze(self) -> FeedbackSummary:
        """Analyze all feedback and return summary."""
        summary = FeedbackSummary(total_entries=len(self.entries))
        if not self.entries:
            return summary

        scores = [e.score for e in self.entries]
        summary.avg_score = sum(scores) / len(scores)

        issue_dist: dict[str, int] = {}
        flagged = set()
        for entry in self.entries:
            entry.issues = self._detect_issues(entry)
            for issue in entry.issues:
                issue_dist[issue] = issue_dist.get(issue, 0) + 1
            if entry.score < 0.5 or "error_in_output" in entry.issues:
                flagged.add(entry.session_id)

        summary.issue_distribution = issue_dist
        summary.low_quality_count = sum(1 for e in self.entries if e.score < 0.5)
        summary.flagged_sessions = sorted(flagged)[:100]

        # Count repair pairs (entries with expected_text)
        summary.repair_pairs = sum(1 for e in self.entries if e.expected_text)

        # Save summary
        atomic_write_json(self.summary_path, asdict(summary))
        return summary

    def get_flagged_sessions(self) -> list[str]:
        """Return session IDs that should be excluded from training."""
        self.analyze()
        return list(dict.fromkeys(entry.session_id for entry in self.entries
                                  if entry.score < 0.3 or "error_in_output" in entry.issues))[:100]

    def generate_repair_pairs(self, output_path: str) -> int:
        """Generate contrastive repair pairs from feedback."""
        pairs = []
        for entry in self.entries:
            if entry.expected_text and entry.output_text != entry.expected_text:
                pairs.append({
                    "input": entry.input_text,
                    "bad_output": entry.output_text,
                    "good_output": entry.expected_text,
                    "issues": entry.issues,
                    "score": entry.score,
                })
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        return len(pairs)

    def summary(self) -> dict:
        s = self.analyze()
        return asdict(s)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Feedback loop analysis")
    parser.add_argument("--analyze", action="store_true", help="Analyze feedback")
    parser.add_argument("--repair-pairs", default=None, help="Generate repair pairs to path")
    parser.add_argument("--flagged", action="store_true", help="List flagged sessions")
    args = parser.parse_args(argv)

    cfg = load()
    collector = FeedbackCollector(cfg)

    if args.analyze:
        s = collector.analyze()
        print(f"[feedback] Total: {s.total_entries}, Avg score: {s.avg_score:.3f}")
        print(f"[feedback] Low quality: {s.low_quality_count}")
        print(f"[feedback] Issues: {s.issue_distribution}")
        print(f"[feedback] Flagged sessions: {len(s.flagged_sessions)}")
    if args.repair_pairs:
        n = collector.generate_repair_pairs(args.repair_pairs)
        print(f"[feedback] Generated {n} repair pairs")
    if args.flagged:
        flagged = collector.get_flagged_sessions()
        for f in flagged[:20]:
            print(f"  {f}")
    if not any([args.analyze, args.repair_pairs, args.flagged]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

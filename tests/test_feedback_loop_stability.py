"""Feedback control regression tests: no network, no production dataset writes."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.feedback_loop import FeedbackCollector, FeedbackEntry


class FakeConfig:
    def __init__(self, root):
        self.root = root
    def path(self, name):
        assert name == "analysis_dir"
        return str(self.root)


class FeedbackStabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="lfm3-feedback-unit-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = FakeConfig(self.root)
        self.collector = FeedbackCollector(self.cfg)

    def test_flagged_session_ids_are_strings_stable_across_summaries(self):
        self.collector.record("session-low", "synthetic-v1", "fabricated",
                              "failed", score=0.1, expected_text="good")
        self.collector.record("session-low", "synthetic-v1", "fabricated2",
                              "failed again", score=0.2)
        self.collector.record("session-ok", "synthetic-v1", "fabricated3",
                              "this is a normal fabricated response", score=0.9)
        before = self.collector.analyze()
        again = self.collector.analyze()
        self.assertEqual(before.issue_distribution, again.issue_distribution)
        self.assertEqual(self.collector.get_flagged_sessions(), ["session-low"])
        self.assertTrue(all(isinstance(s, str) for s in self.collector.get_flagged_sessions()))
        self.assertEqual(again.low_quality_count, 2)
        reopened = FeedbackCollector(self.cfg)
        self.assertEqual(len(reopened.entries), 3)
        self.assertEqual(reopened.get_flagged_sessions(), ["session-low"])

    def test_issue_detector_deduplicates_reanalysis_and_preserves_explicit(self):
        entry = FeedbackEntry(
            entry_id="fixture", session_id="test", model_version="model",
            input_text="fictional", output_text="error",
            score=0.1, issues=["explicit", "too_short"]
        )
        once = self.collector._detect_issues(entry)
        entry.issues = once
        twice = self.collector._detect_issues(entry)
        self.assertEqual(once, twice)
        self.assertEqual(len(once), len(set(once)))
        self.assertIn("error_in_output", once)
        self.assertIn("explicit", once)

    def test_invalid_scores_do_not_enter_training_feedback(self):
        for score in (float("nan"),float("inf"),-0.1,1.1,"0.9",True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.collector.record("bad","synthetic","input","output",score=score)
        self.assertEqual(self.collector.entries, [])

    def test_repair_pairs_are_explicit_and_persist_in_scratch(self):
        self.collector.record("a","synthetic","fiction","incorrect",
                              score=0.2,expected_text="correct")
        self.collector.record("b","synthetic","fiction","correct",
                              score=0.9,expected_text="correct")
        outfile = self.root / "repair.jsonl"
        self.assertEqual(self.collector.generate_repair_pairs(str(outfile)),1)
        pairs=[json.loads(s) for s in outfile.read_text().splitlines()]
        self.assertEqual(pairs[0]["good_output"],"correct")
        self.assertEqual(pairs[0]["bad_output"],"incorrect")
        self.assertEqual(self.collector.summary()["repair_pairs"],2)


if __name__=="__main__":
    unittest.main()

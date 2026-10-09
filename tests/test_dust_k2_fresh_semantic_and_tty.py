"""Fresh semantic benchmark science and terminal display safety tests.

All examples synthetic and public; no K2 prompt corpus, receiver secrets,
network requests, remote shells or model weights loaded by these tests.
"""
import unittest
from unittest.mock import patch

from experiments.dust.k2_fresh_semantic_challenge import (
    CASES, THRESHOLDS, validate_frozen_challenge, score_frozen_cosines)
from experiments.dust.k2_local_semantic_review_session import (
    display_safe_prompt, refuse_remote_reviewer_session, record_decisions)
from test_dust_k2_private_semantic_review import synthetic_queue


class FreshSemanticTests(unittest.TestCase):
    def test_exact_frozen_40_pairs_and_no_old_challenge_leak(self):
        self.assertEqual(len(CASES), 40)
        self.assertEqual(len({s for triple in CASES for s in triple}), 120)
        self.assertEqual(validate_frozen_challenge(),
                         "20609409c5540e1f1b7facba75038ed2ee37bc8aa8e8e252ed9cce0fe7ff5419")
        self.assertEqual(THRESHOLDS, (0.75, 0.85, 0.92))

    def test_frozen_acceptance_and_failure_have_no_training_authority(self):
        passing = score_frozen_cosines([(.9,.4)] * 40)
        failed = score_frozen_cosines([(.8,.88)] * 40)
        self.assertTrue(passing["frozen_threshold_results"]["0.85"]["prespecified_acceptance"])
        self.assertFalse(failed["frozen_threshold_results"]["0.85"]["prespecified_acceptance"])
        self.assertEqual(failed["matched_topic_rank_wins"],0)
        self.assertFalse(passing["case_labels_independently_human_adjudicated"])
        self.assertFalse(passing["classifier_training_authorized"])

    def test_refuse_invalid_cosine_and_population(self):
        for scores in ([], [(.7,.4)] * 39,
                       [(float("nan"),.2)] * 40,
                       [(.8,1.1)] * 40):
            with self.subTest(n=len(scores)):
                with self.assertRaises(ValueError):
                    score_frozen_cosines(scores)


class TerminalSafetyTests(unittest.TestCase):
    def test_injected_esc_bidi_newline_and_zero_width_escape(self):
        dangerous="Normal\x1b]52;c;STEAL\x07\nOther\u202e\u200b"
        safe=display_safe_prompt(dangerous)
        self.assertIn("\\u001b",safe)
        self.assertIn("\\u0007",safe)
        self.assertIn("\\u000a",safe)
        self.assertIn("\\u202e",safe)
        self.assertIn("\\u200b",safe)
        self.assertNotIn("\x1b",safe)
        self.assertNotIn("\u202e",safe)
        self.assertNotIn("\n",safe)
        self.assertIn("Normal",safe)

    def test_oversized_prompt_denied_not_truncated(self):
        for value in ("", "A"*4097, None):
            with self.subTest(v=type(value).__name__):
                with self.assertRaises(ValueError):
                    display_safe_prompt(value)

    def test_detected_remote_session_denied_even_with_tty(self):
        with patch("sys.stdin.isatty",return_value=True), \
             patch("sys.stdout.isatty",return_value=True), \
             patch.dict("os.environ",{"SSH_CONNECTION":"1 2 3 4"}):
            with self.assertRaisesRegex(PermissionError,"remote SSH"):
                refuse_remote_reviewer_session()

    def test_tty_guard_denies_headless_before_prompt_open(self):
        with patch("sys.stdin.isatty",return_value=False):
            with self.assertRaisesRegex(PermissionError,"human TTY"):
                refuse_remote_reviewer_session()

    def test_record_receipt_sanitizes_prompt_but_never_contains_text(self):
        queue=synthetic_queue()
        queue["_source_sha256_from_file"]="e"*64
        lookup={"1"*64:"safe\x1b[2Jtext\nnext",
                "2"*64:"second synthetic", "3"*64:"third synthetic"}
        outputs=[]
        decisions=iter(["s","u"])
        receipt=record_decisions(
            queue,lookup,reviewer_id="researcher-A",
            read=lambda _:next(decisions),write=outputs.append)
        self.assertTrue(any("\\u001b" in x for x in outputs))
        self.assertFalse(any("\x1b" in x for x in outputs))
        self.assertNotIn("safe",str(receipt))
        self.assertEqual(len(receipt["decisions"]),2)


if __name__ == "__main__":
    unittest.main()

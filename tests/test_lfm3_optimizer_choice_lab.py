"""No-model test of constrained LFM3 optimizer-choice expansion."""
import unittest
from src.lfm3_optimizer_choice_lab import CHOICES, admit_choice, choice_messages
from src.lfm3_optimizer_guard import OptimizerResearchRefused, observer_view
from tests.test_lfm3_optimizer_advisor_lab import baseline


class ConstrainedOptimizerTests(unittest.TestCase):
    def test_all_six_options_have_bounded_new_run_proposals(self):
        parent = baseline()
        for letter in CHOICES:
            result = admit_choice(letter, parent)
            self.assertEqual(result["execution_scope"], "new-synthetic-experiment-only")
            self.assertLessEqual(result["learning_rate"], parent["learning_rate"] * 2)
            self.assertEqual(result["steps"], parent["steps"])

    def test_malformed_and_external_labels_refused(self):
        for letter in ("", "STOP", "A;rm /nas", "G", "a", None):
            with self.subTest(letter=letter), self.assertRaises(OptimizerResearchRefused):
                admit_choice(letter, baseline())

    def test_prompt_is_metric_only_not_heldout(self):
        prompts = choice_messages(observer_view(baseline()))
        self.assertIn("synthetic", prompts[1]["content"].lower())
        self.assertNotIn("train_session_sha256", prompts[1]["content"])
        self.assertNotIn("heldout_loss", prompts[1]["content"])
        self.assertIn("Output only", prompts[0]["content"])
        with self.assertRaises(OptimizerResearchRefused):
            choice_messages({"schema":"invalid"})


if __name__ == "__main__":
    unittest.main()

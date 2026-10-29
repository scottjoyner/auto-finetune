"""Second-cycle synthetic optimization observer admission without model load."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_optimizer_feedback_lab import summarize_grid, grid_messages
from src.lfm3_optimizer_guard import OptimizerResearchRefused


def fixture():
    path = Path(__file__).resolve().parents[1] / "research" / "results" / "lfm3_optimizer_six_arm_toy_grid_20261027.json"
    return json.loads(path.read_text())


class FeedbackTests(unittest.TestCase):
    def test_fixed_grid_admits_dev_only_and_selects_measured_best(self):
        base, rows, best = summarize_grid(fixture())
        self.assertEqual(best, "C")
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[2]["optimizer"], "adamw")
        self.assertEqual(rows[2]["learning_rate"], 0.002)
        text = grid_messages(base, rows)[-1]["content"]
        self.assertNotIn("train_session_sha256", text)
        self.assertNotIn("dev_session_sha256", text)
        self.assertNotIn("heldout_loss", text)

    def test_hallucinated_success_or_dataset_overlap_denied(self):
        corrupted = copy.deepcopy(fixture())
        corrupted["heldout_evaluated"] = True
        with self.assertRaises(OptimizerResearchRefused):
            summarize_grid(corrupted)
        corrupted = copy.deepcopy(fixture())
        corrupted["trials"][4]["dev_session_sha256"] = "f"*64
        with self.assertRaises(OptimizerResearchRefused):
            summarize_grid(corrupted)
        corrupted = copy.deepcopy(fixture())
        corrupted["trials"][3]["optimizer"] = "free-text-shell"
        with self.assertRaises(OptimizerResearchRefused):
            summarize_grid(corrupted)


if __name__ == "__main__":
    unittest.main()

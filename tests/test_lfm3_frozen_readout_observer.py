"""Receipt / model boundary tests for read-only LFM3 adapter observer."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_frozen_readout_observer import (
    OptimizerResearchRefused, summarize_readout, observer_messages
)


def fixture():
    path = (Path(__file__).resolve().parents[1] / "research" / "results"
            / "lfm3_real_frozen_readout_adamw_pair_xwing_20261028.json")
    return json.loads(path.read_text())


class FrozenReadoutObserverTests(unittest.TestCase):
    def test_readout_dev_result_is_bound_without_source_data(self):
        view, best = summarize_readout(fixture())
        self.assertEqual(best, "B")
        self.assertEqual(len(view["arms"]), 2)
        self.assertLess(view["arms"][1]["dev_loss"], view["arms"][0]["dev_loss"])
        message = observer_messages(view)[-1]["content"]
        self.assertNotIn("synthetic_train_sha256", message)
        self.assertNotIn("synthetic_dev_sha256", message)
        self.assertNotIn("heldout_loss", message)

    def test_untrusted_accuracy_and_authority_injection_denied(self):
        for key, value in (
            ("production_ready", True),
            ("pretrained_model_modified", True),
            ("verified_train_dev_source_authentication", True),
        ):
            bad = copy.deepcopy(fixture())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(OptimizerResearchRefused):
                summarize_readout(bad)
        bad = copy.deepcopy(fixture())
        bad["arms"][1]["dev_loss_after"] = float("nan")
        with self.assertRaises(OptimizerResearchRefused):
            summarize_readout(bad)
        bad = copy.deepcopy(fixture())
        bad["arms"][0]["dev_loss_before"] = 9999.0
        with self.assertRaises(OptimizerResearchRefused):
            summarize_readout(bad)
        with self.assertRaises(OptimizerResearchRefused):
            observer_messages({"schema": "sg-lfm3-readout-dev-view.v1", "may_execute": True})


if __name__ == "__main__":
    unittest.main()

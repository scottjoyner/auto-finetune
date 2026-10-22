"""No network or model load: policy regressions for xwing boot trainer."""
import unittest

from src.training_admission import Decision, evaluate, MIN_KIB

EMBEDDING = [{"type": "embedding", "loaded_instances": [{"id": "embed"}]}]
LARGE_MODEL = EMBEDDING + [{"type": "llm", "loaded_instances": [{"id": "ornith"}]}]


class AdmissionTests(unittest.TestCase):
    def call(self, **changes):
        args = dict(host="xwing", stopped=False, available_kib=MIN_KIB + 1024,
                    some=0.0, full=0.0, models=EMBEDDING)
        args.update(changes)
        return evaluate(**args)

    def test_safe_host_resources_admitted(self):
        self.assertEqual(self.call(), Decision(True, "admitted"))

    def test_operator_stop_file_overrides_everything(self):
        self.assertEqual(self.call(stopped=True).reason, "stop_file")

    def test_loaded_ornith_denied_even_when_idle(self):
        self.assertEqual(self.call(models=LARGE_MODEL).reason, "resident_inference")

    def test_unknown_inventory_is_not_empty_success(self):
        for inventory in (None, [], [{"type": "llm"}], [{"type": "unexpected", "loaded_instances": []}]):
            with self.subTest(inventory=inventory):
                self.assertFalse(self.call(models=inventory).allowed)

    def test_memory_headroom_required(self):
        for available in (None, -1, 0, MIN_KIB - 1):
            with self.subTest(available=available):
                self.assertEqual(self.call(available_kib=available).reason, "insufficient_memory")

    def test_psi_pressure_gates(self):
        for change in (dict(some=3.1), dict(full=1.1), dict(full=None), dict(some=None),
                       dict(full=-1.0), dict(some=float("nan"))):
            with self.subTest(change=change):
                self.assertEqual(self.call(**change).reason, "memory_pressure")

    def test_other_fleet_nodes_unchanged(self):
        self.assertEqual(self.call(host="x1-370", models=None, available_kib=0),
                         Decision(True, "other_host"))

if __name__ == "__main__":
    unittest.main()

class IntegrationGateTests(unittest.TestCase):
    def test_cli_refuses_before_configuration_or_resource_leases(self):
        from unittest.mock import patch
        from src import cli
        with patch("src.training_admission.read_live", return_value=Decision(False, "resident_inference")):
            with patch.object(cli, "load", side_effect=AssertionError("configuration read before HOLD")):
                self.assertEqual(cli.main(["cli", "train"]), 78)
                self.assertEqual(cli.main(["cli", "dpo"]), 78)

    def test_direct_train_main_refuses_before_dataset_read(self):
        from unittest.mock import patch
        from src import train
        with patch("src.training_admission.read_live", return_value=Decision(False, "stop_file")):
            self.assertEqual(train.main(cfg=None), 78)

    def test_queue_guard_precedes_reset_and_training(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1] / "launch-next.sh").read_text()
        self.assertGreaterEqual(text.count("-m src.training_admission"), 2)
        self.assertLess(text.index("-m src.training_admission"), text.index("rm -f \"$STATE_FILE\""))
        self.assertLess(text.rindex("-m src.training_admission"), text.index("if ! gpu_ok"))

"""Alert event-log replay must preserve acknowledgments without duplication."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.alerting import AlertManager


class FakeConfig:
    def __init__(self, path):
        self.root = path
    def path(self, key):
        assert key == "analysis_dir"
        return str(self.root)
    def get(self, *args, **kwargs):
        return ""


class AlertReplayTests(unittest.TestCase):
    def test_repeated_alerts_are_distinct_and_ack_survives_restart(self):
        with tempfile.TemporaryDirectory(prefix="alerting-replay-") as scratch:
            cfg = FakeConfig(Path(scratch))
            with patch.object(AlertManager, "_notify"):
                first = AlertManager(cfg)
                one = first.fire("warning", "synthetic", "fictional warning")
                two = first.fire("warning", "synthetic", "fictional warning")
                self.assertNotEqual(one, two)
                self.assertTrue(first.acknowledge(one))
                self.assertEqual(len(first.alerts), 2)
                reopened = AlertManager(cfg)
                self.assertEqual(reopened.summary()["total"], 2)
                self.assertEqual(reopened.summary()["unacknowledged"], 1)
                self.assertEqual([a.alert_id for a in reopened.unacknowledged()], [two])
                self.assertTrue(reopened.acknowledge(two))
                final = AlertManager(cfg)
                self.assertEqual(final.summary()["total"], 2)
                self.assertEqual(final.summary()["unacknowledged"], 0)
                self.assertEqual(final.summary()["by_level"]["warning"], 2)

    def test_missing_ack_does_not_modify_log(self):
        with tempfile.TemporaryDirectory(prefix="alerting-missing-") as scratch:
            cfg = FakeConfig(Path(scratch))
            manager = AlertManager(cfg)
            self.assertFalse(manager.acknowledge("nonexistent"))
            self.assertEqual(manager.recent(), [])
            self.assertEqual(manager.summary()["total"], 0)


if __name__ == "__main__":
    unittest.main()

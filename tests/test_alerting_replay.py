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

    def test_notifications_are_mocked_and_log_persists(self):
        import json
        with tempfile.TemporaryDirectory(prefix="alerting-mocked-notify-") as scratch:
            cfg = FakeConfig(Path(scratch))
            cfg.get = lambda *args, **kwargs: "https://example.invalid/fictional"
            with patch("src.alerting.urllib.request.urlopen") as send, \
                    patch("subprocess.run") as desktop:
                manager = AlertManager(cfg)
                alert_id = manager.fire(
                    "critical", "synthetic", "fictional alert",
                    {"origin": "unit-test"})
                send.assert_called_once()
                request = send.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(payload["alert_id"], alert_id)
                self.assertEqual(payload["metadata"]["origin"], "unit-test")
                self.assertEqual(send.call_args.kwargs["timeout"], 5)
                desktop.assert_called_once()
                self.assertIn("fictional alert",
                              (Path(scratch) / "alerts.log").read_text())
            with patch("src.alerting.urllib.request.urlopen", side_effect=OSError("fake failure")), \
                    patch("subprocess.run", side_effect=OSError("fake desktop error")):
                manager.fire("warning", "synthetic", "fictional failure")
            self.assertEqual(len(AlertManager(cfg).alerts), 2)
            self.assertEqual(len(manager.recent(level="critical")), 1)

    def test_missing_ack_does_not_modify_log(self):
        with tempfile.TemporaryDirectory(prefix="alerting-missing-") as scratch:
            cfg = FakeConfig(Path(scratch))
            manager = AlertManager(cfg)
            self.assertFalse(manager.acknowledge("nonexistent"))
            self.assertEqual(manager.recent(), [])
            self.assertEqual(manager.summary()["total"], 0)


if __name__ == "__main__":
    unittest.main()

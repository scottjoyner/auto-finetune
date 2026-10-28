"""Direct registry activation requires affirmative caller confirmation.

Tests isolate all writes in TemporaryDirectory. An explicit keyword flag is
not authenticated approval: this guard only prevents accidental invocation.
"""
import tempfile
import unittest
from pathlib import Path

from src.model_registry import ModelRegistry


class ScratchConfig:
    def __init__(self, directory):
        self.directory = directory
    def path(self, key):
        assert key == "analysis_dir"
        return str(self.directory)


class RegistryActivationGateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="lfm3-registry-unit-")
        self.addCleanup(tmp.cleanup)
        self.registry = ModelRegistry(ScratchConfig(Path(tmp.name)))
        self.base = self.registry.register("baseline","/tmp/fake-base","synthetic-base")
        self.candidate = self.registry.register("candidate","/tmp/fake-candidate","synthetic-base")

    def test_direct_promotion_denied_by_default(self):
        r = self.registry
        self.assertFalse(r.promote(self.candidate))
        self.assertIsNone(r.active_version())
        self.assertTrue(r.promote(self.base, operator_confirmed=True))
        self.assertEqual(r.active_version().version_id, self.base)
        self.assertFalse(r.promote(self.candidate))
        self.assertEqual(r.active_version().version_id, self.base)

    def test_ab_preserves_active_baseline_and_requires_confirmation(self):
        r = self.registry
        r.promote(self.base, operator_confirmed=True)
        self.assertIsNone(r.start_ab_test(self.base,self.base))
        test_id = r.start_ab_test(self.base,self.candidate)
        self.assertIsNotNone(test_id)
        self.assertEqual(r.active_version().version_id,self.base)
        self.assertEqual(r.get(self.candidate).status,"testing")
        self.assertTrue(r.record_ab_metric(test_id,self.base,"score",0.4))
        self.assertTrue(r.record_ab_metric(test_id,self.candidate,"score",0.9))
        self.assertIsNone(r.complete_ab_test(test_id))
        self.assertEqual(r.get_ab_test(test_id).status,"running")
        self.assertEqual(r.active_version().version_id,self.base)
        self.assertEqual(
            r.complete_ab_test(test_id, operator_confirmed=True),self.candidate)
        self.assertEqual(r.active_version().version_id,self.candidate)

    def test_nonfinite_and_reserved_metrics_rejected(self):
        r = self.registry
        tid = r.start_ab_test(self.base,self.candidate)
        for score in (float("nan"),float("inf"),-0.1,1.5,True,"0.5"):
            with self.subTest(score=score):
                self.assertFalse(r.record_ab_metric(tid,self.base,"score",score))
        self.assertFalse(r.record_ab_metric(tid,self.base,"score_count",500))
        self.assertFalse(r.record_ab_metric(tid,"unauthorized","score",0.3))
        self.assertIsNone(r.complete_ab_test(tid,operator_confirmed=True))
        self.assertEqual(r.get_ab_test(tid).status,"running")


if __name__ == "__main__":
    unittest.main()

"""No internal automatic model activation, including obsolete config flags."""
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from src.auto_promote import AutoPromoter, PromotionConfig

class NoAutoActivationTests(unittest.TestCase):
    def test_corrupted_review_state_is_not_silently_reset(self):
        tmp=tempfile.TemporaryDirectory(prefix="pending-review-corruption-")
        self.addCleanup(tmp.cleanup)
        cfg=MagicMock()
        cfg.path.return_value=tmp.name
        state_file=__import__("pathlib").Path(tmp.name)/"auto-promote-state.json"
        for content in ("broken {", "{}", '{"active_ab_test": true, "last_promotion": 0, "last_rollback": 0}'):
            with self.subTest(contents=content):
                state_file.write_text(content)
                with self.assertRaises((ValueError, TypeError)):
                    AutoPromoter(cfg)
                self.assertEqual(state_file.read_text(),content)
        state_file.write_text(json.dumps({
            "active_ab_test":"fictional-review", "last_promotion":0,
            "last_rollback":0}))
        promoter=AutoPromoter(cfg)
        self.assertEqual(promoter.state["active_ab_test"],"fictional-review")

    def test_legacy_config_does_not_bypass_manual_review(self):
        tmp=tempfile.TemporaryDirectory(prefix="activation-boundary-research-")
        self.addCleanup(tmp.cleanup)
        cfg=MagicMock()
        cfg.path.return_value=tmp.name
        p=AutoPromoter(cfg,PromotionConfig(
            min_feedback_samples=1,
            ab_test_duration_hours=0.0,
            require_manual_approval=False))
        p.registry.promote=MagicMock()
        p.registry.complete_ab_test=MagicMock()
        p.state["active_ab_test"]="synthetic-ab"
        p.registry.get_ab_test=MagicMock(return_value=SimpleNamespace(
            status="running", start_time=time.time()-3600,
            version_a="old",version_b="research",
            metrics_a={"score_count":4,"score":1.0},
            metrics_b={"score_count":4,"score":3.6}))
        outcome=p.check_promotion()
        self.assertEqual(outcome["action"],"review_required")
        p.registry.promote.assert_not_called()
        p.registry.complete_ab_test.assert_not_called()
        self.assertEqual(p.state["active_ab_test"],"synthetic-ab")

if __name__=="__main__":unittest.main()

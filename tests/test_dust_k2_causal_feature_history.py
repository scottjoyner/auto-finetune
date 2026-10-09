"""Regression tests for causal, batchwise pre-probe history."""
import unittest
from experiments.dust.k2_causal_feature_history import (
    CausalProbeHistory, FEATURES, SCHEMA,
)

class CausalHistoryTests(unittest.TestCase):
    def make_state(self):
        return CausalProbeHistory(
            expected_population=8, episode_hmac_sha256="a"*64)

    def test_before_first_probe_history_is_unknown_not_future_label(self):
        state=self.make_state()
        self.assertEqual(state.preview(before_candidate=0), (0.,)*8)
        self.assertEqual(len(FEATURES), 8)
        with self.assertRaisesRegex(ValueError,"prior PRE preview"):
            state.commit_batch(first_index=0,clean_loss=1.,
                               plus_losses=[.9,1.1],minus_losses=[1.1,.9],sigma=.25)

    def test_batch_history_unchanged_until_complete_plus_minus_post(self):
        state=self.make_state()
        early=[state.preview(before_candidate=i) for i in range(4)]
        self.assertEqual(early,[(0.,)*8]*4)
        state.commit_batch(first_index=0,clean_loss=1.,
                           plus_losses=[.9,1.2,.8,1.1],
                           minus_losses=[1.1,.8,1.2,.9],sigma=.25)
        later=[state.preview(before_candidate=i) for i in range(4,8)]
        self.assertEqual(later,[later[0]]*4)
        self.assertEqual(later[0][0],.5)
        self.assertEqual(later[0][1],.5)
        self.assertAlmostEqual(later[0][2],0.)
        self.assertAlmostEqual(later[0][7],-.1)
        state.commit_batch(first_index=4,clean_loss=1.,
                           plus_losses=[1.,1.,1.,1.],
                           minus_losses=[1.,1.,1.,1.],sigma=.25)
        self.assertEqual(state.completed_count,8)
        self.assertTrue(state.snapshot()["no_training_authority"])
        self.assertEqual(state.snapshot()["schema"],SCHEMA)

    def test_no_future_commit_duplicate_or_nonfinite(self):
        state=self.make_state()
        state.preview(before_candidate=0)
        with self.assertRaisesRegex(ValueError,"without prior PRE"):
            state.commit_batch(first_index=0,clean_loss=1.,
                               plus_losses=[.9,1.1],minus_losses=[1.1,.9],
                               sigma=.25)
        state.preview(before_candidate=1)
        with self.assertRaises(ValueError):
            state.commit_batch(first_index=0,clean_loss=1.,
                               plus_losses=[float("nan"),1.],
                               minus_losses=[1.,1.],sigma=.25)
        self.assertEqual(state.completed_count,0)
        state.commit_batch(first_index=0,clean_loss=1.,
                           plus_losses=[.9,1.1],minus_losses=[1.1,.9],
                           sigma=.25)
        with self.assertRaisesRegex(ValueError,"out-of-order"):
            state.commit_batch(first_index=0,clean_loss=1.,
                               plus_losses=[1.],minus_losses=[1.],sigma=.25)

    def test_unrelated_source_episode_never_shares_history(self):
        first=self.make_state()
        first.preview(before_candidate=0)
        first.commit_batch(first_index=0,clean_loss=1.,
                           plus_losses=[.8],minus_losses=[1.2],sigma=.25)
        second=CausalProbeHistory(
            expected_population=8, episode_hmac_sha256="b"*64)
        self.assertEqual(second.preview(before_candidate=0),(0.,)*8)
        self.assertNotEqual(first.snapshot()["source_group_hmac_sha256"],
                            second.snapshot()["source_group_hmac_sha256"])

if __name__=="__main__":
    unittest.main()

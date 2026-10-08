"""No-provider, zero-weight T0 training-environment contract."""
import unittest
from scripts.tunix_jev_t0 import SyntheticProbeUpdate, probe_dependencies, run


class T0Tests(unittest.TestCase):
    def test_dependencies_are_read_only(self):
        info = probe_dependencies()
        self.assertIn("tunix", info)
        self.assertIn("jax", info)

    def test_deterministic_two_step_environment(self):
        a = SyntheticProbeUpdate()
        b = SyntheticProbeUpdate()
        probe_a = a.step({"op": "probe", "coordinate": 2})
        probe_b = b.step({"op": "probe", "coordinate": 2})
        self.assertEqual(probe_a, probe_b)
        end = a.step({"op": "update", "coordinate": 2, "sign": 1})
        self.assertTrue(end["done"])
        self.assertFalse(end["production_dispatch_authorized"])
        self.assertEqual(end["provider_tokens"], 0)

    def test_invalid_and_arbitrary_actions_denied(self):
        e = SyntheticProbeUpdate()
        for action in ({"op": "dispatch", "provider": "jev"},
                       {"op": "probe", "coordinate": 100}):
            with self.assertRaises(ValueError):
                e.step(action)
        e.step({"op": "probe", "coordinate": 1})
        with self.assertRaises(ValueError):
            e.step({"op": "route", "coordinate": 1, "sign": 1})
        self.assertEqual(e.step({"op": "abstain"})["reward"], 0.0)

    def test_terminal_episode_fenced(self):
        e = SyntheticProbeUpdate()
        e.step({"op": "probe", "coordinate": 0})
        e.step({"op": "abstain"})
        with self.assertRaises(ValueError):
            e.step({"op": "abstain"})

    def test_bounded_report_without_actual_training(self):
        result = run()
        self.assertFalse(result["Tunix_agent_training_executed"])
        self.assertFalse(result["real_model_weights_read"])
        self.assertEqual(result["hosted_calls"], 0)
        self.assertEqual(len(result["episodes"]), 2)

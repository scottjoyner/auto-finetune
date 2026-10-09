"""CPU/stdlib regression for self-consistent but causally invalid PRE history."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_causal_evidence_verify import verify
from experiments.dust.k2_causal_feature_history import CausalProbeHistory, SCHEMA
from experiments.dust.k2_direction_witness import EVENT_SCHEMA, sha256_json
from experiments.dust.predictive_probe_contract import SCHEMA as LABEL_SCHEMA


def fixture(folder):
    episode = "a" * 64
    model = "b" * 64
    history = CausalProbeHistory(expected_population=8,
                                 episode_hmac_sha256=episode)
    events = []
    labels = []
    prior = "0" * 64

    def append(event):
        nonlocal prior
        event["schema"] = EVENT_SCHEMA
        event["previous_event_sha256"] = prior
        event["event_sha256"] = sha256_json(event)
        prior = event["event_sha256"]
        events.append(event)

    for start in (0, 4):
        pres = []
        plus_scores = []
        minus_scores = []
        for index in range(start, start + 4):
            pre = {
                "phase": "PRE",
                "episode_hmac_sha256": episode,
                "candidate_index": index,
                "features_pre_probe": [0.1] * 8,
                "history_feature_schema": SCHEMA,
                "history_features_pre_probe":
                    list(history.preview(before_candidate=index)),
                "clean_pre_probe": 1.0,
                "sigma": .25,
                "local_monotonic_ns": 10000 + start * 100 + index,
            }
            append(pre)
            pres.append(pre)
        for index in range(start, start + 4):
            p = .9 if index % 2 == 0 else 1.2
            m = 1.1 if index % 2 == 0 else .8
            plus_scores.append(p)
            minus_scores.append(m)
            append({
                "phase": "POST",
                "candidate_index": index,
                "pre_event_sha256": pres[index - start]["event_sha256"],
                "loss_plus": p,
                "loss_minus": m,
                "local_monotonic_ns": 20000 + start * 100 + index,
            })
            labels.append({
                "schema": LABEL_SCHEMA,
                "episode_hmac_sha256": episode,
                "model_revision_sha256": model,
                "candidate_index": index,
                "features_pre_probe": [0.1] * 8,
                "sigma": .25, "loss_clean": 1.0,
                "loss_plus": p, "loss_minus": m,
                "probe_time_order_attested": True,
            })
        history.commit_batch(
            first_index=start, clean_loss=1.,
            plus_losses=plus_scores, minus_losses=minus_scores, sigma=.25)
    summary = {
        "population": 8,
        "causal_history_v2_observed": True,
        "causal_history_v2_training_authorized": False,
        "base_weights_unchanged": True,
        "adapter_weights_unchanged": True,
        "optimizer_updates": 0, "backward_calls": 0,
        "source_episode_hmac_sha256": episode,
        "model_revision_sha256": model,
    }
    paths = [Path(folder) / name for name in (
        "events.jsonl", "labels.jsonl", "summary.json")]
    paths[0].write_text("".join(json.dumps(x) + "\n" for x in events))
    paths[1].write_text("".join(json.dumps(x) + "\n" for x in labels))
    paths[2].write_text(json.dumps(summary))
    return paths, events, labels, summary


class CausalReplayTests(unittest.TestCase):
    def test_valid_complete_pre_post_history_and_label_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, *_ = fixture(tmp)
            result = verify(*paths)
            self.assertEqual(result["observed_directions"], 8)
            self.assertEqual(result["beneficial_plus_directions"], 4)
            self.assertTrue(result["history_pre_only_recomputed"])
            self.assertTrue(result["original_label_schema_preserved"])
            self.assertFalse(result["receiver_signing_key_isolation_accepted"])
            self.assertFalse(result["classifier_training_authorized"])

    def test_rehashed_future_history_still_fails_causal_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, _, _ = fixture(tmp)
            events[8]["history_features_pre_probe"][0] = 0.875
            mapping = {}
            previous = "0" * 64
            for event in events:
                old = event.pop("event_sha256")
                if event["phase"] == "POST":
                    event["pre_event_sha256"] = mapping[event["pre_event_sha256"]]
                event["previous_event_sha256"] = previous
                event["event_sha256"] = sha256_json(event)
                mapping[old] = event["event_sha256"]
                previous = event["event_sha256"]
            paths[0].write_text("".join(json.dumps(x) + "\n" for x in events))
            with self.assertRaisesRegex(ValueError, "PRE history contains"):
                verify(*paths)

    def test_tampered_direction_loss_and_optimizer_promotion_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, events, labels, summary = fixture(tmp)
            labels[0]["loss_plus"] += .2
            paths[1].write_text("".join(json.dumps(x) + "\n" for x in labels))
            with self.assertRaisesRegex(ValueError, "derived classifier label"):
                verify(*paths)
            paths, _, _, summary = fixture(tmp)
            summary["optimizer_updates"] = 1
            paths[2].write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "summary gates"):
                verify(*paths)


if __name__ == "__main__":
    unittest.main()

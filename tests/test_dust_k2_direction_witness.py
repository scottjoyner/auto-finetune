"""Real observer integration tests with a tiny synthetic final-o_proj model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
from types import SimpleNamespace

from experiments.dust.k2_direction_witness import (
    LocalProbeWitness, pseudonym, read_private_key, sha256_json,
)
from experiments.dust.k2_tail_replay import (
    cache_base_sample, init_lora, tail_scored_structured_estimate,
)


def torch_equal_exact(a, b):
    import torch
    return torch.equal(a, b)


class TestWitness(unittest.TestCase):
    def make_toy(self):
        import torch

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.pre = torch.nn.Linear(8, 12, bias=False)
                self.self_attn.o_proj = torch.nn.Linear(12, 8, bias=False)
                self.post_attention_layernorm = torch.nn.LayerNorm(8)
                self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 16),
                                               torch.nn.GELU(),
                                               torch.nn.Linear(16, 8))

            def forward(self, x):
                attention = self.self_attn.o_proj(self.self_attn.pre(x))
                h = x + attention
                return h + self.mlp(self.post_attention_layernorm(h))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer(), Layer()])
                self.model.norm = torch.nn.LayerNorm(8)
                self.lm_head = torch.nn.Linear(8, 32, bias=False)

            def forward(self, input_ids, use_cache=False, return_dict=True):
                x = self.embedding(input_ids)
                for layer in self.model.layers:
                    x = layer(x)
                return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))

        torch.manual_seed(211)
        model = Model().eval().requires_grad_(False)
        sample = {"tokens": [1, 3, 4, 5, 6, 7, 8, 2],
                  "labels": [-100, -100, 4, 5, 6, 7, 8, 2]}
        cache = cache_base_sample(model, sample, "cpu", keep_logits=False)
        a, b = init_lora(cache, rank=4, seed=29)
        return model, cache, a, b, sample

    def test_observed_and_unobserved_estimator_are_identical(self):
        import torch
        model, cache, a, b, sample = self.make_toy()
        before = tail_scored_structured_estimate(
            model, cache, a, b, seed=42, population=8,
            sigma=.25, direction_batch=4)
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "events.jsonl"
            completed = Path(directory) / "completed.jsonl"
            episode = pseudonym(b"z" * 32, sample)
            witness = LocalProbeWitness(
                events, completed, episode_hmac_sha256=episode,
                model_revision_sha256="a" * 64, sigma=.25)
            after = tail_scored_structured_estimate(
                model, cache, a, b, seed=42, population=8,
                sigma=.25, direction_batch=4,
                probe_observer=witness)
            summary = witness.finish(8)
            witness.close()
            self.assertTrue(torch.equal(before["estimate"], after["estimate"]))
            self.assertTrue(torch.equal(before["clean"], after["clean"]))
            self.assertEqual(before["tail_forward_calls"],
                             after["tail_forward_calls"])
            self.assertEqual(before["tokens"], after["tokens"])
            self.assertEqual(summary["rows"], 8)
            self.assertEqual(summary["episodes"], 1)
            self.assertFalse(summary["independent_timing_witness"])
            self.assertFalse(summary["classification_training_authorized"])
            self.assertEqual(stat.S_IMODE(events.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(completed.stat().st_mode), 0o600)
            self.assertEqual(len(events.read_text().splitlines()), 16)
            self.assertEqual(len(completed.read_text().splitlines()), 8)
            self.assertNotIn("[1,3,4,5,6,7,8,2]", events.read_text())
            self.assertNotIn("tokens", completed.read_text())
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(torch.equal(a, a))
            prior = "0" * 64
            seen_pre = set()
            for line in events.read_text().splitlines():
                event = json.loads(line)
                self.assertEqual(event.pop("event_sha256"),
                                 sha256_json(event))
                self.assertEqual(event["previous_event_sha256"], prior)
                prior = sha256_json(event)
                if event["phase"] == "PRE":
                    seen_pre.add(event["event_sha256"]
                                 if "event_sha256" in event else prior)
                else:
                    self.assertIn(event["pre_event_sha256"], seen_pre)

    def test_independent_hmac_receipt_joins_pre_post_before_forward(self):
        from experiments.dust.k2_receipt_receiver import (
            prepare_receiver, receive, verify_event_join)
        model, cache, a, b, sample = self.make_toy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "receiver-owned"
            prepare_receiver(root)
            ev = Path(directory) / "run.events"
            derived = Path(directory) / "run.derived"
            run_id = "a" * 32
            remote_callback = lambda batch_digest: receive(
                root, run_id, batch_digest)
            witness = LocalProbeWitness(
                ev, derived,
                episode_hmac_sha256=pseudonym(b"z" * 32, sample),
                model_revision_sha256="b" * 64,
                sigma=.25, receiver=remote_callback)
            plain = tail_scored_structured_estimate(
                model, cache, a, b, seed=7, population=8,
                sigma=.25, direction_batch=4)
            observed = tail_scored_structured_estimate(
                model, cache, a, b, seed=7, population=8,
                sigma=.25, direction_batch=4,
                probe_observer=witness)
            report = witness.finish(8)
            self.assertEqual(report["receiver_precommit_receipts"], 2)
            self.assertTrue(torch_equal_exact(plain["estimate"],
                                              observed["estimate"]))
            joined = verify_event_join(
                root, run_id, ev, derived, "b" * 64)
            self.assertTrue(joined["derived_label_join_verified"])
            self.assertEqual(joined["derived_label_file_sha256"],
                             hashlib.sha256(derived.read_bytes()).hexdigest())
            self.assertEqual(joined["signed_batch_receipts"], 2)
            self.assertEqual(joined["completed_direction_count"], 8)
            self.assertTrue(joined["all_post_scores_follow_receiver_ack"])
            self.assertFalse(joined["classifier_training_authorized"])
            broken_labels = Path(directory) / "tampered.derived"
            forged = [json.loads(row) for row in
                      derived.read_text().splitlines()]
            forged[0]["loss_plus"] += 0.25
            broken_labels.write_text("\n".join(
                json.dumps(row) for row in forged) + "\n")
            with self.assertRaisesRegex(ValueError, "derived label differs"):
                verify_event_join(root, run_id, ev, broken_labels, "b" * 64)
            tampered = Path(directory) / "tampered.events"
            tampered.write_text(ev.read_text().replace('"phase":"RECEIPT"',
                                                        '"phase":"BAD_PHASE"', 1))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_event_join(root, run_id, tampered)

    def test_private_episode_key_requires_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory) / "key"
            key.write_bytes(b"q" * 32)
            key.chmod(0o644)
            with self.assertRaises(PermissionError):
                read_private_key(key)
            key.chmod(0o600)
            self.assertEqual(read_private_key(key), b"q" * 32)
            first = {"tokens": [1, 2, 3, 4],
                     "labels": [-100, -100, 3, 4]}
            response_variant = {"tokens": [1, 2, 7, 8, 9],
                                "labels": [-100, -100, 7, 8, 9]}
            self.assertNotEqual(pseudonym(b"q" * 32, first),
                                pseudonym(b"r" * 32, first))
            # The same masked prompt must remain one episode group even
            # when target response text/length differs.
            self.assertEqual(pseudonym(b"q" * 32, first),
                             pseudonym(b"q" * 32, response_variant))
            with self.assertRaisesRegex(ValueError, "masked prompt"):
                pseudonym(b"q" * 32, {"tokens": [3], "labels": [3]})

    def test_no_overwrite_and_incomplete_pre_probe_fails(self):
        model, cache, a, b, sample = self.make_toy()
        with tempfile.TemporaryDirectory() as directory:
            ev, done = (Path(directory) / name
                        for name in ("events.jsonl", "done.jsonl"))
            w = LocalProbeWitness(ev, done, episode_hmac_sha256="a" * 64,
                                  model_revision_sha256="b" * 64,
                                  sigma=.25)
            with self.assertRaises(FileExistsError):
                LocalProbeWitness(ev, done,
                                  episode_hmac_sha256="a" * 64,
                                  model_revision_sha256="b" * 64,
                                  sigma=.25)
            import torch
            d = torch.ones((1, 8))
            clean = torch.ones((1, 2))
            pos = torch.tensor([1, 2])
            w.before_batch(directions=d, clean=clean, cache=cache,
                           positions=pos, first=0)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                w.finish(1)
            with self.assertRaisesRegex(RuntimeError, "previous probe"):
                w.before_batch(directions=d, clean=clean, cache=cache,
                               positions=pos, first=1)
            w.close()

    def test_explicit_runner_optin(self):
        from experiments.dust.k2_real_direction_probe import main
        with self.assertRaises(SystemExit) as error:
            main(["--model-dir", "/nothing", "--train-jsonl", "/nothing",
                  "--expected-sha256", "a" * 64,
                  "--episode-key-file", "/nothing", "--events", "/nothing1",
                  "--derived", "/nothing2", "--output", "/nothing3"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

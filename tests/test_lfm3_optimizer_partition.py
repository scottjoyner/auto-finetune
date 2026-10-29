"""Metadata-only LFM3 optimizer partition, no PyTorch/model load."""
import unittest
from src.lfm3_optimizer_partition import MATRIX, plan_full_model_metadata
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused


class MockParam:
    def __init__(self, shape):
        self.shape = shape


class OptimizerPartitionTests(unittest.TestCase):
    def small_inventory(self):
        return [
            ("model.layers.0.feed_forward.w1.weight", MockParam((128, 128))),
            ("model.layers.2.self_attn.q_proj.weight", MockParam((64, 64))),
            ("model.layers.3.conv.in_proj.weight", MockParam((64, 64))),
            ("model.layers.2.self_attn.gate_proj.weight", MockParam((16, 128))),
            ("model.embed_tokens.weight", MockParam((8192, 256))),
            ("model.norm.weight", MockParam((256,))),
        ]

    def test_full_model_review_is_mandatory(self):
        with self.assertRaises(OptimizerResearchRefused):
            plan_full_model_metadata(self.small_inventory(),
                                     model_class="Lfm3ForCausalLM",
                                     checkpoint_sha256=CHECKPOINT)
        with self.assertRaises(OptimizerResearchRefused):
            plan_full_model_metadata(self.small_inventory(),
                                     model_class="OtherModel",
                                     checkpoint_sha256=CHECKPOINT,
                                     explicit_research_review=True)

    def test_explicit_matrix_allowlist_not_every_matrix(self):
        x = plan_full_model_metadata(
            self.small_inventory(), model_class="Lfm3ForCausalLM",
            checkpoint_sha256=CHECKPOINT, explicit_research_review=True
        )
        self.assertEqual(x["candidate_muon_tensor_count"], 2)
        self.assertEqual(x["candidate_muon_parameters"], 128*128+64*64)
        self.assertEqual(x["category_tensors"]["adamw_conv_and_conv_projection"], 1)
        self.assertEqual(x["category_tensors"]["adamw_head_gate_projection"], 1)
        self.assertFalse(x["actual_optimizer_allocated"])
        self.assertFalse(x["muon_eligibility_reviewed_for_training"])

    def test_bad_shapes_aliases_or_layers_rejected(self):
        p = MockParam((2048, 2048))
        x = [
            ("model.layers.25.self_attn.q_proj.weight", p),
            ("model.embed_tokens.weight", MockParam((8192, 256))),
        ]
        with self.assertRaises(OptimizerResearchRefused):
            plan_full_model_metadata(x, model_class="Lfm3ForCausalLM",
                                     checkpoint_sha256=CHECKPOINT,
                                     explicit_research_review=True)
        x = self.small_inventory()
        x.append(("model.layers.5.feed_forward.w2.weight", x[0][1]))
        with self.assertRaises(OptimizerResearchRefused):
            plan_full_model_metadata(x, model_class="Lfm3ForCausalLM",
                                     checkpoint_sha256=CHECKPOINT,
                                     explicit_research_review=True)
        self.assertIsNotNone(MATRIX.fullmatch("model.layers.0.feed_forward.w1.weight"))
        self.assertIsNone(MATRIX.fullmatch("model.layers.0.conv.in_proj.weight"))


if __name__ == "__main__":
    unittest.main()

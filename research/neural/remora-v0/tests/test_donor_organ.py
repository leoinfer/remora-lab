import unittest

import torch

from remora.donors.neural_ir import qwen_gated_delta_core_ir, qwen_shared_expert_ir
from remora.donors.payload import QwenSharedExpertOrgan, functional_equivalence, reference_shared_expert


class DonorOrganTests(unittest.TestCase):
    def _tensors(self):
        return {
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "model.language_model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, 8, dtype=torch.bfloat16),
        }

    def test_standalone_organ_matches_independent_reference(self):
        tensors = self._tensors()
        hidden = torch.randn(2, 3, 8)
        organ = QwenSharedExpertOrgan(tensors)
        actual = organ(hidden)
        expected = reference_shared_expert(hidden, tensors)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_functional_equivalence_records_stateless_contract(self):
        tensors = self._tensors()
        result = functional_equivalence(tensors, batch=1, time=1)
        self.assertLess(result["relative_l2_error"], 1e-6)
        self.assertIsNone(result["state_update_error"])

    def test_ir_validates_qwen_shared_expert(self):
        ir = qwen_shared_expert_ir(source_revision="fixture", tensor_names=list(self._tensors()))
        ir.validate()
        self.assertEqual(ir.state_contract["kind"], "stateless")
        self.assertEqual(ir.output_contract["names"], ["output"])

    def test_ir_validates_minimal_gdn_core(self):
        ir = qwen_gated_delta_core_ir(
            component_id="fixture-gdn-core",
            source_revision="fixture",
            tensor_names=["q_weight", "k_weight", "v_weight", "a_weight", "b_weight", "conv_qkv", "A_log", "dt_bias"],
            layer=17,
            value_head=10,
            input_width=96,
        )
        ir.validate()
        self.assertEqual(ir.output_contract["names"], ["core"])
        self.assertEqual(ir.state_contract["owned_state"]["recurrent"], ["batch", 128, 128])
        self.assertEqual(ir.metadata["excluded_coadapted_tensors"], ["z_weight", "norm_weight", "out_weight"])


if __name__ == "__main__":
    unittest.main()

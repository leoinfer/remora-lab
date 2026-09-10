import unittest

import torch

from remora.donors.graft import LowRankPort, QwenSharedExpertGraft, make_donor_variant


class DonorGraftTests(unittest.TestCase):
    def _tensors(self):
        return {
            "x.shared_expert.gate_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "x.shared_expert.up_proj.weight": torch.randn(4, 8, dtype=torch.bfloat16),
            "x.shared_expert.down_proj.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "x.shared_expert_gate.weight": torch.randn(1, 8, dtype=torch.bfloat16),
        }

    def test_low_rank_port_has_explicit_parameter_count(self):
        port = LowRankPort(6, 10, rank=3)
        self.assertEqual(port.parameter_count, 3 * (6 + 10))

    def test_variant_controls_preserve_shapes_without_mutating_source(self):
        tensors = self._tensors()
        original = {name: value.clone() for name, value in tensors.items()}
        shuffled = make_donor_variant(tensors, "shuffled", seed=9)
        random = make_donor_variant(tensors, "random", seed=9)
        zero = make_donor_variant(tensors, "zero", seed=9)
        for name in tensors:
            self.assertEqual(shuffled[name].shape, tensors[name].shape)
            self.assertEqual(random[name].shape, tensors[name].shape)
            self.assertTrue(torch.equal(zero[name], torch.zeros_like(tensors[name])))
            self.assertTrue(torch.equal(tensors[name], original[name]))

    def test_graft_uses_bus_contract_and_only_ports_are_trainable(self):
        graft = QwenSharedExpertGraft(self._tensors(), bus_dim=6, donor_dim=8, rank=2)
        output = graft(torch.randn(2, 3, 6))
        self.assertEqual(output.shape, (2, 3, 6))
        self.assertEqual(graft.port_parameter_count, 2 * 2 * (6 + 8))
        self.assertTrue(all(parameter.requires_grad for parameter in graft.trainable_port_parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in graft.organ.parameters()))
        self.assertTrue(graft.interface_signature()["donor_core_frozen"])

    def test_variant_controls_have_identical_tensor_geometry(self):
        tensors = self._tensors()
        variants = [make_donor_variant(tensors, name, seed=3) for name in ("actual", "shuffled", "random", "zero")]
        for name in tensors:
            self.assertEqual({variant[name].shape for variant in variants}, {tensors[name].shape})


if __name__ == "__main__":
    unittest.main()

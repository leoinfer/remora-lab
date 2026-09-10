import unittest

import torch

from remora.donors.gdn import QwenGatedDeltaCoreOrgan, QwenGatedDeltaCoreSocket, QwenGatedDeltaHead, TrainableGatedDeltaCoreOrgan, causal_depthwise_silu
from remora.donors.router import (
    QwenCompactRouterOrgan,
    QwenRouterOrgan,
    make_router_variant,
    router_functional_metrics,
    select_balanced_pair,
    top_right_singular_basis,
)


class DonorRouterTests(unittest.TestCase):
    def test_rank_basis_is_orthonormal_and_has_expected_geometry(self):
        weight = torch.randn(8, 16)
        basis = top_right_singular_basis(weight, 6)
        self.assertEqual(tuple(basis.shape), (16, 6))
        self.assertTrue(torch.allclose(basis.transpose(0, 1) @ basis, torch.eye(6), atol=1e-5, rtol=1e-5))

    def test_identity_basis_reproduces_selected_router_rows(self):
        weight = torch.randn(8, 16)
        identity = torch.eye(16)
        pair = select_balanced_pair(weight, identity)
        organ = QwenRouterOrgan(weight, input_basis=identity, selected_rows=pair)
        inputs = torch.randn(2, 3, 16)
        expected = inputs @ weight[list(pair)].transpose(0, 1)
        self.assertTrue(torch.allclose(organ(inputs), expected, atol=1e-6, rtol=1e-6))
        metrics = router_functional_metrics(weight, identity, selected_rows=pair, inputs=inputs)
        self.assertLess(metrics["relative_l2_error"], 1e-6)

    def test_core_controls_keep_geometry_and_repairs_are_the_only_trainable_state(self):
        weight = torch.randn(8, 16)
        basis = top_right_singular_basis(weight, 6)
        pair = select_balanced_pair(weight, basis)
        actual = QwenRouterOrgan(weight, input_basis=basis, selected_rows=pair, trainable_repair_rank=1)
        random = QwenRouterOrgan(
            weight,
            input_basis=basis,
            selected_rows=pair,
            donor_variant="random",
            variant_seed=4,
            trainable_repair_rank=1,
        )
        self.assertEqual(actual.donor_parameter_count, random.donor_parameter_count)
        self.assertEqual(actual.trainable_parameter_count, 1 * (6 + 16))
        self.assertTrue(all(not value.requires_grad for value in actual.buffers()))
        self.assertTrue(all(value.requires_grad for value in actual.port_parameters()))
        self.assertFalse(torch.equal(actual.donor_weight, random.donor_weight))

    def test_router_variants_do_not_mutate_source(self):
        weight = torch.randn(8, 16)
        original = weight.clone()
        for variant in ("actual", "random", "shuffled", "zero"):
            value = make_router_variant(weight, variant, seed=12)
            self.assertEqual(tuple(value.shape), tuple(weight.shape))
        self.assertTrue(torch.equal(weight, original))

    def test_warm_start_repair_is_a_zero_function_with_live_first_step_gradient(self):
        weight = torch.randn(8, 16)
        basis = top_right_singular_basis(weight, 6)
        organ = QwenRouterOrgan(
            weight,
            input_basis=basis,
            selected_rows=None,
            trainable_repair_rank=1,
            repair_initialization="warm_start",
        )
        inputs = torch.randn(2, 3, 6)
        with torch.no_grad():
            initial = organ(inputs)
        self.assertTrue(torch.allclose(initial, inputs @ (weight @ basis).T, atol=1e-5, rtol=1e-5))
        loss = organ(inputs).square().mean()
        loss.backward()
        self.assertGreater(float(organ.repair_up.weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(organ.repair_down.weight.grad.abs().sum()), 0.0)

    def test_compact_router_uses_transformed_core_and_matched_repair(self):
        compact = torch.randn(8, 6)
        organ = QwenCompactRouterOrgan(
            compact,
            trainable_repair_rank=1,
            repair_initialization="warm_start",
        )
        inputs = torch.randn(2, 3, 6)
        with torch.no_grad():
            self.assertTrue(torch.allclose(organ(inputs), inputs @ compact.T, atol=1e-6, rtol=1e-6))
        self.assertEqual(organ.donor_parameter_count, compact.numel())
        self.assertEqual(organ.trainable_parameter_count, 6 + 8)

    def test_gdn_head_is_stateful_and_causal(self):
        torch.manual_seed(3)
        kwargs = {
            "q_weight": torch.randn(128, 8),
            "k_weight": torch.randn(128, 8),
            "v_weight": torch.randn(128, 8),
            "z_weight": torch.randn(128, 8),
            "a_weight": torch.randn(1, 8),
            "b_weight": torch.randn(1, 8),
            "conv_qkv": torch.randn(384, 2),
            "A_log": torch.tensor([0.0]),
            "dt_bias": torch.tensor([-1.0]),
            "norm_weight": torch.ones(128),
            "out_weight": torch.randn(6, 128),
        }
        organ = QwenGatedDeltaHead(**kwargs)
        inputs = torch.randn(2, 4, 8)
        output, state = organ(inputs, return_state=True)
        self.assertEqual(tuple(output.shape), (2, 4, 6))
        self.assertEqual(tuple(state[0].shape), (2, 128, 128))
        self.assertEqual(tuple(state[1].shape), (2, 384, 1))
        changed = inputs.clone()
        changed[:, -1] += 5.0
        first = organ(changed, return_state=False)[:, :-1]
        self.assertTrue(torch.allclose(first, output[:, :-1], atol=1e-5, rtol=1e-5))
        self.assertEqual(tuple(causal_depthwise_silu(inputs.new_zeros(2, 4, 384), kwargs["conv_qkv"]).shape), (2, 4, 384))
        left, left_state = organ(inputs[:, :2], return_state=True)
        right, right_state = organ(inputs[:, 2:], state=left_state, return_state=True)
        self.assertTrue(torch.allclose(torch.cat((left, right), dim=1), output, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(right_state[0], state[0], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(right_state[1], state[1], atol=1e-5, rtol=1e-5))

    def test_gdn_core_socket_exports_frozen_core_and_bounded_repair(self):
        torch.manual_seed(9)
        kwargs = {
            "q_weight": torch.randn(128, 8),
            "k_weight": torch.randn(128, 8),
            "v_weight": torch.randn(128, 8),
            "z_weight": torch.randn(128, 8),
            "a_weight": torch.randn(1, 8),
            "b_weight": torch.randn(1, 8),
            "conv_qkv": torch.randn(384, 2),
            "A_log": torch.tensor([0.0]),
            "dt_bias": torch.tensor([-1.0]),
            "norm_weight": torch.ones(128),
            "out_weight": torch.randn(8, 128),
        }
        organ = QwenGatedDeltaHead(**kwargs)
        socket = QwenGatedDeltaCoreSocket(organ, d_model=192, repair_rank=2)
        hidden = torch.randn(2, 4, 192)
        output = socket(hidden)
        self.assertEqual(tuple(output.shape), (2, 4, 192))
        self.assertEqual(tuple(socket.last_core.shape), (2, 4, 128))
        self.assertTrue(torch.allclose(output[..., :128], socket.last_core, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(output[..., 128:], torch.zeros_like(output[..., 128:]), atol=1e-6))
        self.assertEqual(socket.trainable_parameter_count, 2 * (192 + 8 + 128 + 192))
        self.assertTrue(all(not parameter.requires_grad for parameter in socket.organ.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in socket.input_port.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in socket.port_parameters()))

    def test_gdn_core_organ_matches_full_head_recurrent_core(self):
        torch.manual_seed(10)
        kwargs = {
            "q_weight": torch.randn(128, 8),
            "k_weight": torch.randn(128, 8),
            "v_weight": torch.randn(128, 8),
            "z_weight": torch.randn(128, 8),
            "a_weight": torch.randn(1, 8),
            "b_weight": torch.randn(1, 8),
            "conv_qkv": torch.randn(384, 2),
            "A_log": torch.tensor([0.0]),
            "dt_bias": torch.tensor([-1.0]),
            "norm_weight": torch.ones(128),
            "out_weight": torch.randn(8, 128),
        }
        head = QwenGatedDeltaHead(**kwargs)
        core = QwenGatedDeltaCoreOrgan(
            q_weight=kwargs["q_weight"],
            k_weight=kwargs["k_weight"],
            v_weight=kwargs["v_weight"],
            a_weight=kwargs["a_weight"],
            b_weight=kwargs["b_weight"],
            conv_qkv=kwargs["conv_qkv"],
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
        )
        inputs = torch.randn(2, 4, 8)
        _, _, head_details = head(inputs, return_state=True, return_intermediates=True)
        core_output, _, core_details = core(inputs, return_state=True, return_intermediates=True)
        self.assertTrue(torch.allclose(core_output, head_details["core"], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(core_details["value"], head_details["value"], atol=1e-6, rtol=1e-6))
        self.assertEqual(core.donor_parameter_count, 3858)

    def test_fresh_gdn_core_is_trainable_same_geometry_control(self):
        torch.manual_seed(11)
        kwargs = {
            "q_weight": torch.randn(128, 8),
            "k_weight": torch.randn(128, 8),
            "v_weight": torch.randn(128, 8),
            "a_weight": torch.randn(1, 8),
            "b_weight": torch.randn(1, 8),
            "conv_qkv": torch.randn(384, 2),
            "A_log": torch.tensor([0.0]),
            "dt_bias": torch.tensor([-1.0]),
        }
        fresh = TrainableGatedDeltaCoreOrgan(**kwargs)
        self.assertEqual(fresh.donor_parameter_count, 3858)
        self.assertEqual(fresh.trainable_parameter_count, 3858)
        self.assertEqual(len(list(fresh.buffers())), 0)
        self.assertTrue(all(parameter.requires_grad for parameter in fresh.parameters()))

    def test_core_socket_accepts_native_width_fixed_projection(self):
        torch.manual_seed(12)
        core = QwenGatedDeltaCoreOrgan(
            q_weight=torch.randn(128, 2560),
            k_weight=torch.randn(128, 2560),
            v_weight=torch.randn(128, 2560),
            a_weight=torch.randn(1, 2560),
            b_weight=torch.randn(1, 2560),
            conv_qkv=torch.randn(384, 2),
            A_log=torch.tensor([0.0]),
            dt_bias=torch.tensor([-1.0]),
        )
        projection = torch.randn(2560, 192)
        socket = QwenGatedDeltaCoreSocket(core, d_model=192, input_projection=projection)
        output = socket(torch.randn(2, 3, 192))
        self.assertEqual(tuple(output.shape), (2, 3, 192))
        self.assertIsNone(socket.interface_permutation)
        self.assertEqual(socket.interface_signature()["input_projection_kind"], "fixed_analytic")
        self.assertTrue(all(not parameter.requires_grad for parameter in socket.input_port.parameters()))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from remora.config import ModelConfig
from remora.models import build_model
from remora.modules.recurrent import GatedDeltaState
from remora.modules import SwiGLUExpert
from remora.utils import count_parameters


class CoreModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.cfg = ModelConfig(
            vocab_size=128, max_seq_len=16, d_model=32, n_layers=2,
            n_heads=4, d_ff=32, bus_dim=16, n_experts=2, adapter_dim=4,
            baseline_d_ff=64,
        )

    def test_forward_backward_and_replacement(self):
        model = build_model("remora", self.cfg)
        x = torch.randint(0, 128, (3, 16))
        y = torch.randint(0, 128, (3, 16))
        logits, loss, aux = model(x, y, return_aux=True)
        self.assertEqual(logits.shape, (3, 16, 128))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertTrue(torch.allclose(model.blocks[0].plastic.up.weight, torch.zeros_like(model.blocks[0].plastic.up.weight)))
        model.replace_expert(0, 0, SwiGLUExpert(self.cfg.bus_dim, self.cfg.d_ff, self.cfg.bus_dim))
        logits, loss = model(x, y)
        self.assertEqual(logits.shape[-1], 128)

    def test_baseline_is_same_order_of_magnitude(self):
        remora = build_model("remora", self.cfg)
        baseline = build_model("baseline", self.cfg)
        ratio = count_parameters(baseline) / count_parameters(remora)
        self.assertLess(ratio, 3.0)
        self.assertGreater(ratio, 0.3)

    def test_direct_bus_ablation_preserves_forward_contract(self):
        model = build_model("remora", self.cfg, bus_mode="direct")
        x = torch.randint(0, 128, (2, 16))
        logits, loss, aux = model(x, x, return_aux=True)
        self.assertEqual(logits.shape, (2, 16, 128))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.bus.interface_signature()["trainable_parameters"], 0)
        self.assertLess(count_parameters(model), count_parameters(build_model("remora", self.cfg)))

    def test_parallel_recurrent_scan_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/XPU associative scan is unavailable")
        torch.manual_seed(12)
        reference = GatedDeltaState(16, 8, parallel_scan=False).cuda().eval()
        parallel = GatedDeltaState(16, 8, parallel_scan=True).cuda().eval()
        parallel.load_state_dict(reference.state_dict())
        x = torch.randn(3, 11, 16, device="cuda")
        initial = torch.randn(3, 8, device="cuda")
        with torch.inference_mode():
            reference_y, reference_state = reference(x, initial)
            parallel_y, parallel_state = parallel(x, initial)
        self.assertTrue(torch.allclose(reference_y, parallel_y, rtol=2e-4, atol=2e-5))
        self.assertTrue(torch.allclose(reference_state, parallel_state, rtol=2e-4, atol=2e-5))

    def test_parallel_recurrent_scan_backward_matches_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/XPU associative scan is unavailable")
        torch.manual_seed(13)
        reference = GatedDeltaState(16, 8, parallel_scan=False).cuda().train()
        parallel = GatedDeltaState(16, 8, parallel_scan=True).cuda().train()
        parallel.load_state_dict(reference.state_dict())
        reference_x = torch.randn(2, 9, 16, device="cuda", requires_grad=True)
        parallel_x = reference_x.detach().clone().requires_grad_(True)
        reference_initial = torch.randn(2, 8, device="cuda", requires_grad=True)
        parallel_initial = reference_initial.detach().clone().requires_grad_(True)

        reference_y, reference_state = reference(reference_x, reference_initial)
        (reference_y.square().mean() + reference_state.square().mean()).backward()
        parallel_y, parallel_state = parallel(parallel_x, parallel_initial)
        (parallel_y.square().mean() + parallel_state.square().mean()).backward()

        self.assertTrue(torch.allclose(reference_y, parallel_y, rtol=2e-4, atol=2e-5))
        self.assertTrue(torch.allclose(reference_x.grad, parallel_x.grad, rtol=4e-4, atol=4e-5))
        self.assertTrue(torch.allclose(reference_initial.grad, parallel_initial.grad, rtol=4e-4, atol=4e-5))
        for expected, actual in zip(reference.parameters(), parallel.parameters()):
            self.assertTrue(torch.allclose(expected.grad, actual.grad, rtol=4e-4, atol=4e-5))


if __name__ == "__main__":
    unittest.main()

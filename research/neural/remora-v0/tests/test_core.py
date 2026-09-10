import unittest

import torch

from remora.config import ModelConfig
from remora.models import build_model
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


if __name__ == "__main__":
    unittest.main()

import unittest

import torch

from environments.lifetime_curriculum import build_lifetime_curriculum
from remora.config import ModelConfig
from remora.models import build_model
from remora.modules import lora_parameter_names, matching_suffixes, replace_linear_with_lora
from remora.utils import count_parameters, freeze_all


class LoRAAndCurriculumTests(unittest.TestCase):
    def test_lora_replacement_is_zero_initialized_and_accounted(self):
        cfg = ModelConfig(
            vocab_size=128,
            max_seq_len=16,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=32,
            bus_dim=16,
            n_experts=2,
            adapter_dim=4,
            baseline_d_ff=64,
        )
        model = build_model("baseline", cfg)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        selected = replace_linear_with_lora(
            model,
            matching_suffixes(("attention.qkv", "attention.out")),
            rank=2,
        )
        freeze_all(model)
        for name, parameter in model.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                parameter.requires_grad = True
        self.assertEqual(len(selected), 4)
        self.assertEqual(count_parameters(model, trainable_only=True), 2 * 2 * (32 + 96 + 32 + 32))
        self.assertEqual(len(lora_parameter_names(model)), 8)
        x = torch.randn(2, 16, 32)
        for name, old in before.items():
            if name in model.state_dict() and name.endswith(("base.weight", "base.bias")):
                # The wrapper preserves the original base values exactly.
                self.assertTrue(torch.equal(model.state_dict()[name], old))
        self.assertTrue(torch.allclose(model.blocks[0].attention.qkv.lora_B, torch.zeros_like(model.blocks[0].attention.qkv.lora_B)))
        self.assertTrue(torch.isfinite(model.blocks[0].attention.qkv(x)).all())

    @unittest.skipUnless(torch.cuda.is_available(), "accelerator is unavailable")
    def test_dynamic_lora_insertion_inherits_accelerator_device(self):
        cfg = ModelConfig(
            vocab_size=128,
            max_seq_len=8,
            d_model=32,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            bus_dim=16,
            n_experts=2,
            adapter_dim=4,
            baseline_d_ff=64,
        )
        model = build_model("baseline", cfg).cuda()
        replace_linear_with_lora(
            model,
            matching_suffixes(("attention.qkv", "attention.out")),
            rank=2,
        )
        self.assertTrue(all(parameter.device.type == "cuda" for name, parameter in model.named_parameters() if "lora_" in name))
        x = torch.randint(0, 128, (2, 8), device="cuda")
        logits, _ = model(x)
        self.assertEqual(logits.shape, (2, 8, 128))

    def test_curriculum_is_deterministic_and_has_shifted_interfaces(self):
        first, metadata_first = build_lifetime_curriculum(train_count=12, eval_count=8)
        second, metadata_second = build_lifetime_curriculum(train_count=12, eval_count=8)
        self.assertEqual(metadata_first, metadata_second)
        self.assertEqual([task.stage_id for task in first], ["T1", "T2", "T3", "T4", "T5"])
        self.assertEqual([task.concept_id for task in first], ["parity", "parity", "parity", "mod3", "compare"])
        for left, right in zip(first, second):
            self.assertEqual(left.train, right.train)
            self.assertNotEqual({example.prompt for example in left.valid}, {example.prompt for example in left.shifted})
            self.assertNotEqual({example.prompt for example in left.valid}, {example.prompt for example in left.unseen})
        self.assertTrue(all(len(task.shifted) == 8 and len(task.unseen) == 8 for task in first))
        self.assertTrue(all(example.response in {"0", "1", "2"} for task in first for example in task.train))


if __name__ == "__main__":
    unittest.main()

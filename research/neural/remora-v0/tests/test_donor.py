import tempfile
import unittest
from pathlib import Path

import torch

from remora.config import ModelConfig
from remora.donors.manifest import inspect_resident_model
from remora.donors.port import TeacherPortAdapter, teacher_logit_distillation_loss
from remora.donors.registry import DonorRegistry


class DonorTests(unittest.TestCase):
    def test_header_only_manifest_and_compatibility_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"model_type":"foreign","text_config":{"hidden_size":64,"vocab_size":999}}\n')
            (root / "LICENSE").write_text("Example License\n")
            from safetensors.torch import save_file

            save_file({"layer.0.weight": torch.zeros(4, 8)}, str(root / "model.safetensors"))
            manifest = inspect_resident_model(root, ModelConfig(vocab_size=128, d_model=32).to_dict())
            self.assertFalse(manifest["inspection"]["weights_materialized"])
            self.assertFalse(manifest["inspection"]["get_tensor_called"])
            self.assertEqual(manifest["headers"]["parsed_tensor_count"], 1)
            self.assertEqual(manifest["headers"]["tensor_inventory"][0]["shape"], [4, 8])
            self.assertEqual(manifest["compatibility"]["status"], "INCOMPATIBLE_FOR_DIRECT_GRAFT")

    def test_teacher_port_is_trainable_and_versioned(self):
        torch.manual_seed(4)
        port = TeacherPortAdapter(12, 6)
        features = torch.randn(2, 5, 12)
        packet = port(features)
        self.assertEqual(packet.latent.shape, (2, 5, 6))
        self.assertEqual(packet.version, "donor-port-v1")
        packet.latent.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in port.parameters()))

    def test_logit_distillation_rejects_unaligned_vocabularies(self):
        with self.assertRaises(ValueError):
            teacher_logit_distillation_loss(torch.randn(2, 3, 5), torch.randn(2, 3, 7))

    def test_donor_promotion_requires_external_decision(self):
        registry = DonorRegistry()
        donor_id = registry.register_manifest({"path": "/tmp/donor", "files": {}, "tensor_index": {}, "license": {}})
        registry.propose_candidate(donor_id, "candidate", "frozen_teacher", "module-x", "donor-port-v1")
        with self.assertRaises(PermissionError):
            registry.decide("candidate", "PROMOTED", {"score": 1.0})
        candidate = registry.decide("candidate", "FROZEN_EVALUATED", {"score": 1.0}, external_decision=True)
        self.assertEqual(candidate.state, "FROZEN_EVALUATED")


if __name__ == "__main__":
    unittest.main()

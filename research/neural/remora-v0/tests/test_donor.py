import tempfile
import unittest
from pathlib import Path

import torch

from remora.config import ModelConfig
from remora.donors.manifest import inspect_resident_model
from remora.donors.port import TeacherPortAdapter, teacher_logit_distillation_loss
from remora.donors.registry import DonorRegistry
from remora.donors.selection import select_components
from remora.donors.extract import extract_selected_tensors


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

    def test_component_selection_is_bounded_and_value_free(self):
        manifest = {
            "headers": {"tensor_inventory": [
                {"name": "model.layers.0.linear_attn.a", "tensor_class": "recurrent_or_gated_linear_attention", "nbytes": 40},
                {"name": "model.layers.0.linear_attn.b", "tensor_class": "recurrent_or_gated_linear_attention", "nbytes": 40},
                {"name": "model.layers.1.linear_attn.a", "tensor_class": "recurrent_or_gated_linear_attention", "nbytes": 40},
                {"name": "model.layers.0.mlp.experts.a", "tensor_class": "expert_or_mlp", "nbytes": 1000},
            ]},
            "compatibility": {"status": "INCOMPATIBLE_FOR_DIRECT_GRAFT"},
        }
        result = select_components(manifest, ["recurrent_or_gated_linear_attention"], 100)
        self.assertEqual(result["selected_payload_bytes"], 80)
        self.assertEqual(result["selected"][0]["component_key"], "model.layers.0.linear_attn")
        self.assertEqual(len(result["selected"][0]["tensor_names"]), 2)
        self.assertTrue(result["selection_is_value_free"])
        self.assertEqual(result["recommended_import"], "frozen_teacher_or_activation_distillation")

    def test_payload_extraction_requires_opt_in_and_reads_only_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "donor"
            root.mkdir()
            (root / "config.json").write_text('{"model_type":"foreign"}\n')
            from safetensors.torch import save_file

            save_file(
                {"layer.0.weight": torch.arange(8, dtype=torch.float32), "unused": torch.ones(4)},
                str(root / "model.safetensors"),
            )
            manifest = inspect_resident_model(root, ModelConfig(vocab_size=128, d_model=32).to_dict())
            selected = select_components(
                manifest,
                ["other"],
                max_payload_bytes=64,
                max_components=1,
            )
            with self.assertRaises(PermissionError):
                extract_selected_tensors(root, manifest, selected, Path(tmp) / "candidate.safetensors")
            output = Path(tmp) / "candidate.safetensors"
            receipt = extract_selected_tensors(
                root,
                manifest,
                selected,
                output,
                allow_payload=True,
                max_payload_bytes=64,
            )
            self.assertTrue(receipt["payload_materialized"])
            self.assertFalse(receipt["model_loader_called"])
            self.assertEqual(receipt["tensor_names"], ["layer.0.weight"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()

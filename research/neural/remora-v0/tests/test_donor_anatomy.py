import unittest

from remora.donors.anatomy import build_anatomy_graph
from remora.donors.organ import select_anatomy_component


class DonorAnatomyTests(unittest.TestCase):
    def test_groups_packed_and_shared_components_from_headers(self):
        manifest = {
            "path": "/tmp/donor",
            "config": {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
                "text_config": {
                    "hidden_size": 2560,
                    "shared_expert_intermediate_size": 640,
                    "hidden_act": "silu",
                    "num_hidden_layers": 48,
                    "num_experts": 512,
                },
            },
            "headers": {
                "index_name_reconciliation": {"ok": True},
                "tensor_inventory": [
                    {"name": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", "filename": "a.safetensors", "shape": [640, 2560], "dtype": "BF16", "nbytes": 3276800},
                    {"name": "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", "filename": "a.safetensors", "shape": [640, 2560], "dtype": "BF16", "nbytes": 3276800},
                    {"name": "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", "filename": "a.safetensors", "shape": [2560, 640], "dtype": "BF16", "nbytes": 3276800},
                    {"name": "model.language_model.layers.0.mlp.shared_expert_gate.weight", "filename": "a.safetensors", "shape": [1, 2560], "dtype": "BF16", "nbytes": 5120},
                    {"name": "model.language_model.layers.0.mlp.experts.gate_up_proj", "filename": "b.safetensors", "shape": [512, 1280, 2560], "dtype": "BF16", "nbytes": 3355443200},
                    {"name": "model.language_model.layers.0.mlp.experts.down_proj", "filename": "c.safetensors", "shape": [512, 2560, 640], "dtype": "BF16", "nbytes": 3355443200},
                ],
            },
        }
        graph = build_anatomy_graph(manifest, {"d_model": 192, "bus_dim": 96})
        self.assertEqual(graph["component_count"], 2)
        shared = next(item for item in graph["components"] if item["architecture_family"] == "shared_swiglu_expert")
        self.assertEqual(shared["tensor_count"], 4)
        self.assertEqual(shared["payload_bytes"], 9835520)
        self.assertEqual(shared["geometry"]["input_width"], 2560)
        self.assertEqual(shared["geometry"]["output_width"], 2560)
        self.assertFalse(shared["actual_payload_inspected"])
        self.assertEqual(shared["compatibility"]["status"], "WIDTH_MISMATCH_REQUIRES_CONVERSION")

    def test_autopsy_inventory_format_is_accepted(self):
        manifest = {
            "config": {"text_config": {"hidden_size": 8}},
            "tensor_inventory": [
                {"tensor_name": "model.language_model.layers.0.linear_attn.A_log", "shard": "a.safetensors", "shape": [4], "dtype": "BF16", "payload_bytes": 8}
            ],
        }
        graph = build_anatomy_graph(manifest)
        self.assertEqual(graph["inspection"]["parsed_tensor_count"], 1)
        self.assertEqual(graph["components"][0]["architecture_family"], "gated_deltanet_linear_attention")
        self.assertFalse(graph["inspection"]["weights_materialized"])

    def test_organ_selection_is_value_free_and_budgeted(self):
        manifest = {
            "config": {"text_config": {"hidden_size": 16, "hidden_act": "silu", "shared_expert_intermediate_size": 4}},
            "headers": {"tensor_inventory": [
                {"name": "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", "filename": "a", "shape": [4, 16], "dtype": "BF16", "nbytes": 128},
                {"name": "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", "filename": "a", "shape": [4, 16], "dtype": "BF16", "nbytes": 128},
                {"name": "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", "filename": "a", "shape": [16, 4], "dtype": "BF16", "nbytes": 128},
                {"name": "model.language_model.layers.0.mlp.shared_expert_gate.weight", "filename": "a", "shape": [1, 16], "dtype": "BF16", "nbytes": 32},
            ]},
        }
        graph = build_anatomy_graph(manifest, {"d_model": 8, "bus_dim": 4})
        selection = select_anatomy_component(graph, "qwen3.8.language.layer0.shared_expert", max_payload_bytes=512)
        self.assertTrue(selection["selection_is_value_free"])
        self.assertFalse(selection["donor_weights_consulted"])
        self.assertEqual(selection["selected_payload_bytes"], 416)
        with self.assertRaises(ValueError):
            select_anatomy_component(graph, "qwen3.8.language.layer0.shared_expert", max_payload_bytes=415)


if __name__ == "__main__":
    unittest.main()

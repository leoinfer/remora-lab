"""Safe inspection and controlled import ports for resident donor models."""

from .anatomy import build_anatomy_graph
from .manifest import inspect_resident_model
from .organ import select_anatomy_component
from .neural_ir import NeuralComponentIR, NeuralIROp, qwen_gated_delta_core_ir, qwen_gated_delta_head_ir, qwen_shared_expert_ir
from .payload import QwenSharedExpertOrgan, functional_equivalence, inspect_payload, load_payload, reference_shared_expert
from .graft import LowRankPort, QwenSharedExpertGraft, make_donor_variant
from .router import QwenCompactRouterOrgan, QwenRouterOrgan, make_router_variant, select_balanced_pair, top_right_singular_basis
from .gdn import QwenGatedDeltaCoreOrgan, QwenGatedDeltaCoreSocket, QwenGatedDeltaHead, TrainableGatedDeltaCoreOrgan, causal_depthwise_silu, l2_normalize, make_gdn_variant
from .port import TeacherPortAdapter, teacher_logit_distillation_loss
from .registry import DonorRegistry
from .selection import group_components, select_components
from .extract import extract_selected_tensors
from .response import DonorResponseRecord, load_response_records, response_contract_signature
from .activation import (
    DonorActivationRecord,
    activation_contract_signature,
    load_activation_bundle,
    load_activation_records,
    write_activation_bundle,
    write_activation_records,
)
from .runtime import DonorRuntimeSpec, LocalTransformersDonor

__all__ = [
    "build_anatomy_graph",
    "inspect_resident_model",
    "select_anatomy_component",
    "NeuralComponentIR",
    "NeuralIROp",
    "qwen_shared_expert_ir",
    "qwen_gated_delta_core_ir",
    "qwen_gated_delta_head_ir",
    "QwenSharedExpertOrgan",
    "functional_equivalence",
    "inspect_payload",
    "load_payload",
    "reference_shared_expert",
    "LowRankPort",
    "QwenSharedExpertGraft",
    "make_donor_variant",
    "QwenRouterOrgan",
    "QwenCompactRouterOrgan",
    "QwenGatedDeltaHead",
    "QwenGatedDeltaCoreOrgan",
    "QwenGatedDeltaCoreSocket",
    "TrainableGatedDeltaCoreOrgan",
    "causal_depthwise_silu",
    "l2_normalize",
    "make_gdn_variant",
    "make_router_variant",
    "select_balanced_pair",
    "top_right_singular_basis",
    "TeacherPortAdapter",
    "teacher_logit_distillation_loss",
    "DonorRegistry",
    "group_components",
    "select_components",
    "extract_selected_tensors",
    "DonorResponseRecord",
    "load_response_records",
    "response_contract_signature",
    "DonorActivationRecord",
    "activation_contract_signature",
    "load_activation_bundle",
    "load_activation_records",
    "write_activation_bundle",
    "write_activation_records",
    "DonorRuntimeSpec",
    "LocalTransformersDonor",
]

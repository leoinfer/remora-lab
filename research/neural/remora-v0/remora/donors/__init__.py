"""Safe inspection and controlled import ports for resident donor models."""

from .manifest import inspect_resident_model
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
    "inspect_resident_model",
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

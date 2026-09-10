"""Safe inspection and controlled import ports for resident donor models."""

from .manifest import inspect_resident_model
from .port import TeacherPortAdapter, teacher_logit_distillation_loss
from .registry import DonorRegistry
from .selection import group_components, select_components
from .extract import extract_selected_tensors

__all__ = [
    "inspect_resident_model",
    "TeacherPortAdapter",
    "teacher_logit_distillation_loss",
    "DonorRegistry",
    "group_components",
    "select_components",
    "extract_selected_tensors",
]

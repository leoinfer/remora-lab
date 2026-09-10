"""Remora-v0: a small, falsifiable modular neural research instrument."""

from .config import ModelConfig
from .models import BaselineModel, RemoraModel, build_model

__all__ = ["ModelConfig", "RemoraModel", "BaselineModel", "build_model"]

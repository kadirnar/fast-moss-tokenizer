"""Inference-only optimizations. The upstream checkpoint remains unchanged."""

from .loading import load_model
from .v2_loading import load_model as load_model_v2

__all__ = ["load_model", "load_model_v2"]

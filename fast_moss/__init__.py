"""Inference-only optimizations. The upstream checkpoint remains unchanged."""

from .loading import load_model

__all__ = ["load_model"]

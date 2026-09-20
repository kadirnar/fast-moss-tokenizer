"""Fast inference for the native 48 kHz stereo MOSS Audio Tokenizer v2."""

from .v2 import codec, encode_fixed, load_model, optimized
from .graphs import GraphedCallable

load_model_v2 = load_model

__all__ = [
    "load_model",
    "load_model_v2",
    "optimized",
    "codec",
    "encode_fixed",
    "GraphedCallable",
]

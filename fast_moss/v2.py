"""Public API for exact 48 kHz stereo MOSS Audio Tokenizer v2 inference.

Loading, codec adapters, runtime ownership and kernels live in separate modules.
Existing imports from ``fast_moss.v2`` remain stable.
"""

from .v2_codec import codec, encode_fixed
from .v2_loading import MODEL_ID, REVISION, load_model
from .v2_runtime import V2Runtime, optimized, validate

__all__ = [
    "MODEL_ID",
    "REVISION",
    "load_model",
    "optimized",
    "codec",
    "encode_fixed",
    "V2Runtime",
    "validate",
]

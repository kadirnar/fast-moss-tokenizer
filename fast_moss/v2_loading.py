"""Pinned native 48 kHz stereo checkpoint and its original mixed-precision policy."""

import torch
from transformers import AutoModel

from .loading import strict_precision

MODEL_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"


def load_model(device="cuda", *, attention_implementation="sdpa", compute_dtype="bf16"):
    """Keep FP32 checkpoint parameters and the native FP32 quantizer.

    SDPA is explicit for reproducibility: it is also upstream's fallback when
    the optional flash-attn package is absent. BF16 compute is the v2 default.
    """
    if attention_implementation not in ("sdpa", "flash_attention_2"):
        raise ValueError("Unknown v2 attention implementation")
    if compute_dtype not in ("bf16", "fp32"):
        raise ValueError("Unknown v2 compute dtype")
    strict_precision()
    # Preserve parameter version counters even when called from an inference
    # context; cache ownership uses them to detect changed parameter values.
    with torch.inference_mode(False):
        return (
            AutoModel.from_pretrained(
                MODEL_ID,
                revision=REVISION,
                trust_remote_code=True,
                dtype=torch.float32,
                codec_weight_dtype="fp32",
                attention_implementation=attention_implementation,
                compute_dtype=compute_dtype,
            )
            .eval()
            .to(device)
            .requires_grad_(False)
        )

"""Reversible inference transformations, keeping the original FP32 arithmetic."""
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F

from .kernels import scale_add


def _frozen_conv_forward(self, x):
    return self._conv_forward(x, self._fast_weight, self.bias)


def _sa_block(self, x):
    normalized = self.norm1(x)
    update = self.self_attn(normalized, normalized, normalized)
    return self._fast_scale_add(x, update, self.layer_scale_1.scale)


def _ff_block(self, x):
    update = self.linear2(self.activation(self.linear1(self.norm2(x))))
    return self._fast_scale_add(x, update, self.layer_scale_2.scale)


def _decode_latents(self, latents):
    encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
    encodings = F.normalize(encodings)
    dist = (encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ self._fast_codebook.t() + self._fast_codebook_norm)
    indices = (-dist).max(1)[1].reshape(latents.size(0), -1)
    return self.decode_code_wo_out_proj(indices).float(), indices


@contextmanager
def optimized(model, residual_backend="none", cache_codebooks=True, cache_weights=True,
              kv_backend="none", rope_backend="none", share_rope_tables=False,
              attention_mask_backend="none"):
    """Temporarily optimize a frozen model; no parameter conversion or retraining.

    Do not mutate weights or use the same model concurrently inside this context.
    Cached normalization uses the exact upstream operations, once per codebook.
    """
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Call eval().requires_grad_(False) before inference optimization")
    if getattr(model, "_fast_optimization_active", False):
        raise RuntimeError("This model already has an active optimization context")
    if residual_backend not in {"triton", "cute", "none"}:
        raise ValueError("Unknown residual backend")
    if kv_backend not in {"none", "triton"}:
        raise ValueError("Unknown KV backend")
    if rope_backend not in {"none", "triton"}:
        raise ValueError("Unknown RoPE backend")
    if share_rope_tables and rope_backend!="triton":
        raise ValueError("Shared RoPE tables require the Triton RoPE backend")
    if attention_mask_backend not in {"none", "triton"}:
        raise ValueError("Unknown attention mask backend")
    kernel = scale_add
    if residual_backend == "cute":
        from .cute_kernels import scale_add as kernel
    saved = []

    def replace(obj, name, value):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    try:
        replace(model, "_fast_optimization_active", True)
        for module in model.modules():
            kind = type(module).__name__
            if kind == "MossAudioTokenizerTransformer" and (share_rope_tables or attention_mask_backend == "triton"):
                from .transformer import stage_forward
                if len(module.layers):
                    if any(layer.self_attn.weights_per_step for layer in module.layers):
                        raise ValueError("Shared tables do not support per-step weights")
                    replace(module,"_fast_original_stage",module.forward)
                    replace(module,"_fast_share_rope",share_rope_tables)
                    replace(module,"_fast_share_masks",attention_mask_backend == "triton")
                    replace(module,"forward",MethodType(stage_forward,module))
            if kind == "MossAudioTokenizerMultiheadAttention" and attention_mask_backend == "triton":
                from .attention import forward as attention_forward
                if module.weights_per_step or len(module.in_projs) != 1 or len(module.out_projs) != 1:
                    raise ValueError("Fused masks do not support per-step attention weights")
                replace(module, "_fast_mask_pool", None)
                replace(module, "_fast_original_attention", module.forward)
                replace(module, "forward", MethodType(attention_forward, module))
            if kind == "MossAudioTokenizerRotaryEmbedding" and rope_backend == "triton":
                from .rope import forward
                replace(module, "_fast_freqs", {})
                replace(module, "_fast_tables", None)
                replace(module, "_fast_original_rope", module.forward)
                replace(module, "forward", MethodType(forward, module))
            if kind == "MossAudioTokenizerMultiheadAttention" and kv_backend == "triton":
                from .kv_cache import attention_complete
                if module.weights_per_step:
                    raise ValueError("Fused KV updates require per-lane offsets")
                replace(module, "_fast_original_complete", module._complete_kv)
                replace(module, "_complete_kv", MethodType(attention_complete, module))
            if (cache_weights and isinstance(module, torch.nn.Conv1d)
                    and torch.nn.utils.parametrize.is_parametrized(module, "weight")):
                replace(module, "_fast_weight", module.weight.detach())
                replace(module, "forward", MethodType(_frozen_conv_forward, module))
            if kind == "MossAudioTokenizerTransformerLayer" and residual_backend != "none":
                if module.gating is not None or module.weights_per_step:
                    raise ValueError("Unsupported transformer configuration")
                replace(module, "_fast_scale_add", kernel)
                replace(module, "_sa_block", MethodType(_sa_block, module))
                replace(module, "_ff_block", MethodType(_ff_block, module))
            if kind == "MossAudioTokenizerLFQ" and cache_codebooks:
                cb = F.normalize(module.codebook.weight.float())
                replace(module, "_fast_codebook", cb)
                replace(module, "_fast_codebook_norm", cb.pow(2).sum(1, keepdim=True).t())
                replace(module, "decode_latents", MethodType(_decode_latents, module))
        yield model
    finally:
        for obj, name, existed, old in reversed(saved):
            if existed:
                setattr(obj, name, old)
            else:
                delattr(obj, name)

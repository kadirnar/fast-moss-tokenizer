"""Reversible inference transformations, keeping the original FP32 arithmetic."""
from contextlib import contextmanager, ExitStack
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
              attention_mask_backend="none", quantizer_backend="none", matrix_backend="none",
              projection_backend="none", ffn_backend="none", norm_backend="none"):
    """Temporarily optimize a frozen model; no precision conversion or retraining.

    Do not mutate weights or use the same model concurrently inside this context.
    Cached normalization uses the exact upstream operations, once per codebook.
    matrix_backend='cublaslt' explicitly changes weight storage/strides until exit;
    it invalidates managed graphs on this device and requires the bundled profile.
    matrix_backend='triton' additionally uses ordered FP32 kernels for validated
    attention/FFN shapes; other shapes retain the supported cuBLASLt/native dispatch.
    projection_backend='triton' adds eight-channel LFQ projection kernels and a
    64 MiB decoder table; it requires cached weights and the validated environment.
    ffn_backend='triton' uses two stages and decoder epilogues for 24-row FFNs,
    plus fused native one-row GEMV epilogues in both encoder and decoder;
    it requires Triton matrices, residual fusion and the pinned ffn math extra.
    norm_backend='cuda' uses the pinned normalization extra for profiled FP32
    LayerNorm shapes, preserving the native Welford tree and affine arithmetic.
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
    if quantizer_backend not in {"none", "triton"}:
        raise ValueError("Unknown quantizer backend")
    if quantizer_backend == "triton" and not cache_codebooks:
        raise ValueError("Quantizer fusion requires cached codebooks")
    if matrix_backend not in {"none", "cublaslt", "triton"}:
        raise ValueError("Unknown matrix backend")
    if norm_backend not in {"none", "cuda"}:
        raise ValueError("Unknown normalization backend")
    if ffn_backend not in {'none','triton'}:
        raise ValueError('Unknown FFN backend')
    if ffn_backend == 'triton':
        if matrix_backend != 'triton' or residual_backend == 'none':
            raise ValueError('FFN fusion requires Triton matrices and an enabled residual backend')
        from .ffn import math_library
        ffn_library = math_library()
    if projection_backend not in {'none', 'triton'}:
        raise ValueError('Unknown projection backend')
    if projection_backend == 'triton':
        if not cache_weights:
            raise ValueError('Projection kernels require cached convolution weights')
        from .projections import validate
        validate(model)
    kernel = scale_add
    if residual_backend == "cute":
        from .cute_kernels import scale_add as kernel
    saved = []
    original_ffn = {}
    matrix_stack = ExitStack()

    def replace(obj, name, value):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    try:
        replace(model, "_fast_optimization_active", True)
        if matrix_backend in {'cublaslt', 'triton'}:
            from .matrices import MatrixRuntime
            matrix_stack.enter_context(MatrixRuntime(model, backend=matrix_backend))
        if norm_backend == 'cuda':
            from .normalization import NormalizationRuntime
            matrix_stack.enter_context(NormalizationRuntime(model))
        if projection_backend == 'triton':
            from .projections import cache_lifetime
            matrix_stack.enter_context(cache_lifetime(next(model.parameters()).device))
        if quantizer_backend == "triton":
            if type(model).__name__ != "MossAudioTokenizerModel" or type(model.quantizer).__name__ != "MossAudioTokenizerResidualLFQ":
                raise ValueError("Quantizer fusion requires the original MOSS LFQ model")
            from .quantizer import encode_frame
            replace(model, "_fast_original_encode_frame", model._encode_frame)
            replace(model, "_encode_frame", MethodType(encode_frame, model))
        for module in model.modules():
            kind = type(module).__name__
            if kind == 'MossAudioTokenizerTransformerLayer' and ffn_backend == 'triton':
                original_ffn[id(module)] = module._ff_block
            if kind == "MossAudioTokenizerResidualLFQ" and quantizer_backend == "triton":
                from .quantizer import residual_forward
                replace(module, "_fast_codes_only", False)
                replace(module, "forward", MethodType(residual_forward, module))
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
                if quantizer_backend == "triton":
                    from .quantizer import decode_latents, forward
                    replace(module, "decode_latents", MethodType(decode_latents, module))
                    replace(module, "forward", MethodType(forward, module))
        if ffn_backend == 'triton':
            from .ffn import forward as ffn_forward
            runtime = model._fast_matrix_runtime
            owned = {id(module) for module,_ in runtime.saved}
            encoder = {id(module) for module in model.encoder.modules()} if hasattr(model,'encoder') else set()
            for module in model.modules():
                if (type(module).__name__ == 'MossAudioTokenizerTransformerLayer'
                        and id(module.linear1) in owned and id(module.linear2) in owned
                        and tuple(module.linear1.weight.shape) == (5120,1280)
                        and tuple(module.linear2.weight.shape) == (1280,5120)):
                    replace(module,'_fast_original_ffn',module._ff_block)
                    replace(module,'_fast_observed_ffn',original_ffn[id(module)])
                    replace(module,'_fast_ffn_runtime',runtime)
                    replace(module,'_fast_ffn_library',ffn_library)
                    replace(module,'_fast_ffn_fuse',id(module) not in encoder)
                    replace(module,'_ff_block',MethodType(ffn_forward,module))
        if projection_backend == 'triton':
            from .projections import forward as project_forward, decode_codes
            q = model.quantizer
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                table = torch.empty((32, 1024, 512), device=next(model.parameters()).device, dtype=torch.float32)
                for i, quantizer in enumerate(q.quantizers):
                    conv = quantizer.out_proj
                    # Use the original convolution arithmetic once per frozen
                    # entry. Direct functional calls do not emit observer hooks.
                    values = F.conv1d(quantizer.codebook.weight.T[None], conv._fast_weight, conv.bias)
                    table[i].copy_(values[0].T)
                    replace(conv, '_fast_original_projection', conv.forward)
                    replace(conv, 'forward', MethodType(project_forward, conv))
            replace(q, '_fast_projected_codebooks', table)
            replace(q, '_fast_original_decode_codes', q.decode_codes)
            replace(q, 'decode_codes', MethodType(decode_codes, q))
        yield model
    finally:
        try:
            matrix_stack.close()
        finally:
            for obj, name, existed, old in reversed(saved):
                if existed:
                    setattr(obj, name, old)
                else:
                    delattr(obj, name)

"""Exact distance postprocessing and embedding gather around vendor FP32 GEMM."""
import torch
import torch.nn.functional as F
from torch.nn.modules import module as module_hooks
import triton
import triton.language as tl


@triton.jit
def _select(Dots, Norm, CodeNorm, Book, Latents, Out, Indices,
            K: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
            LB: tl.constexpr, LD: tl.constexpr, LT: tl.constexpr,
            STE: tl.constexpr, BLOCK: tl.constexpr, CHANNELS: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    dots = tl.load(Dots + row * K + k, k < K, 0)
    norm = tl.load(Norm + row)
    cnorm = tl.load(CodeNorm + k, k < K, 0)
    # Preserve both FP32 roundings. Dropping the common row norm can change ties.
    distance = (norm - dots) + cnorm
    distance = tl.where(k < K, distance, float('inf'))
    # torch.max(-distance) chooses the first NaN, or first finite minimum.
    nan_index = tl.min(tl.where((distance != distance) & (k < K), k, 2147483647), 0)
    finite = tl.where(distance == distance, distance, float('inf'))
    index = tl.argmin(finite, 0, tie_break_left=True)
    index = tl.where(nan_index != 2147483647, nan_index, index)
    tl.store(Indices + row, index.to(tl.int64))
    d = tl.arange(0, CHANNELS)
    value = tl.load(Book + index * D + d, d < D, 0)
    if STE:
        latent = tl.load(Latents + (row // T) * LB + d * LD + (row % T) * LT, d < D, 0)
        value = latent + (value - latent)
        offset = (row // T) * D * T + d * T + row % T
    else:
        offset = row * D + d
    tl.store(Out + offset, value, d < D)


def select(dots, row_norm, codebook_norm, codebook, latents, *, straight_through=False):
    """Keep GEMM/norm inputs unchanged; fuse distance, index, gather, optional STE.

    Without STE output has the upstream embedding's transposed B,D,T strides.
    With canonical contiguous latents, STE is fused into contiguous B,D,T output.
    Other latent layouts retain the upstream additions and their output strides.
    """
    if latents.ndim != 3 or codebook.ndim != 2:
        raise ValueError('Expected B,D,T latents and K,D codebook')
    b, d, t = latents.shape
    k, book_d = codebook.shape
    if (d != book_d or not k or k > 65536 or dots.shape != (b * t, k)
            or row_norm.shape != (b * t, 1) or codebook_norm.shape != (1, k)):
        raise ValueError('Incompatible quantizer shapes or unsupported codebook size')
    tensors = (dots, row_norm, codebook_norm, codebook, latents)
    if any(x.dtype != torch.float32 or not x.is_cuda or x.device != dots.device for x in tensors):
        raise ValueError('Quantizer fusion requires same-device CUDA FP32 inputs')
    if not all(x.is_contiguous() for x in tensors[:-1]):
        raise ValueError('Distance inputs and codebook must be contiguous')
    if straight_through and latents.stride() != (d * t, t, 1):
        values, indices = select(dots, row_norm, codebook_norm, codebook, latents)
        return latents + (values - latents), indices
    indices = torch.empty((b, t), device=dots.device, dtype=torch.long)
    output = (torch.empty((b, d, t), device=dots.device, dtype=torch.float32) if straight_through
              else torch.empty((b, t, d), device=dots.device, dtype=torch.float32))
    if b * t:
        _select[(b * t,)](dots, row_norm, codebook_norm, codebook, latents, output, indices,
                         k, d, t, *latents.stride(), straight_through,
                         triton.next_power_of_2(k), triton.next_power_of_2(d),
                         num_warps=4, enable_fp_fusion=False)
    return (output if straight_through else output.transpose(1, 2)), indices


def decode_latents(module, latents, *, straight_through=False):
    runtime = getattr(module, '_fast_quantizer_prepare_runtime', None)
    prepared = runtime.prepare_quantizer(module, latents) if runtime is not None else None
    if prepared is None:
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1]).float()
        encodings = F.normalize(encodings)
        row_norm = encodings.pow(2).sum(1, keepdim=True)
        twice = 2 * encodings
    else:
        row_norm, twice = prepared
    dots = twice @ module._fast_codebook.t()
    return select(dots, row_norm, module._fast_codebook_norm, module.codebook.weight,
                  latents, straight_through=straight_through)


def forward(self, z):
    z_e = self.in_proj(z.float()).float()
    z_q, indices = decode_latents(self, z_e, straight_through=True)
    return self.out_proj(z_q).float(), indices, z_e


@triton.jit
def _update(R, U, Q, Mask, NewR, NewQ, Masked, N: tl.constexpr,
            D: tl.constexpr, T: tl.constexpr, SUM: tl.constexpr,
            NEXT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = tl.load(Mask + (i // (D * T)) * T + i % T, i < N, 0).to(tl.float32)
    update = tl.load(U + i, i < N, 0) * mask
    if SUM:
        value = tl.load(Q + i, i < N, 0) + update
        tl.store(NewQ + i, value, i < N)
    if NEXT:
        residual = tl.load(R + i, i < N, 0) - update
        tl.store(NewR + i, residual, i < N)
        tl.store(Masked + i, residual * mask, i < N)


def residual_forward(self, z, input_length, n_quantizers=None):
    """Preserve public quantized outputs; omit them only for the encoder caller."""
    codes_only = self._fast_codes_only
    z = self.input_proj(z).float()
    b, d, t = z.shape
    mask = torch.arange(t, device=z.device).expand(b, t) < input_length.unsqueeze(1)
    mask3 = mask.unsqueeze(1)
    quantized = None if codes_only else torch.zeros_like(z, dtype=torch.float32)
    residual = z.clone().float()
    masked = residual * mask3
    all_indices = []
    count = min(n_quantizers or self.num_quantizers, len(self.quantizers))
    for i, quantizer in enumerate(self.quantizers):
        if i >= count:
            break
        last = i == count - 1
        if codes_only and last:
            # No later residual consumes the final projection; retain the exact
            # normalized distance calculation needed for the final code.
            _, indices = quantizer.decode_latents(quantizer.in_proj(masked.float()).float())
        elif codes_only and hasattr(self, '_fast_projected_codebooks'):
            from .projections import can_fuse_update, project_update
            if can_fuse_update(residual, quantizer):
                latent = quantizer.in_proj(masked.float()).float()
                latent, indices = decode_latents(quantizer, latent, straight_through=True)
                residual, masked = project_update(latent.contiguous(), quantizer.out_proj._fast_weight,
                                                   quantizer.out_proj.bias, residual, mask)
            else:
                update, indices, _ = quantizer(masked)
                residual = residual - update * mask3
                masked = residual * mask3
        else:
            update, indices, _ = quantizer(masked)
            if (residual.is_cuda and residual.dtype == torch.float32 and residual.is_contiguous()
                    and update.dtype == torch.float32 and update.is_contiguous()
                    and (quantized is None or quantized.is_contiguous())):
                new_q = torch.empty_like(quantized) if quantized is not None else None
                new_r = torch.empty_like(residual) if not last else None
                new_masked = torch.empty_like(residual) if not last else None
                _update[(triton.cdiv(residual.numel(), 256),)](
                    residual, update, quantized if quantized is not None else residual, mask,
                    new_r if new_r is not None else residual,
                    new_q if new_q is not None else residual,
                    new_masked if new_masked is not None else residual,
                    residual.numel(), d, t, not codes_only, not last, 256, enable_fp_fusion=False)
                quantized, residual, masked = new_q, new_r, new_masked
            else:
                if not codes_only:
                    quantized = quantized + update * mask3
                if not last:
                    residual = residual - update * mask3
                    masked = residual * mask3
        all_indices.append(indices)
    codes = (torch.stack(all_indices) if all_indices else
             torch.empty(0, b, t, device=z.device, dtype=torch.long))
    return (None if codes_only else self.output_proj(quantized)), codes, input_length


def encode_frame(self, *args, **kwargs):
    # The pinned encoder discards quantized_out. Hooks can observe that value or
    # the final projection, so retain the complete path when any are installed.
    if (module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
            or any(m._forward_hooks or m._forward_pre_hooks for m in self.quantizer.modules())):
        return self._fast_original_encode_frame(*args, **kwargs)
    previous = self.quantizer._fast_codes_only
    self.quantizer._fast_codes_only = True
    try:
        return self._fast_original_encode_frame(*args, **kwargs)
    finally:
        self.quantizer._fast_codes_only = previous

"""Exact distance postprocessing and embedding gather around vendor FP32 GEMM."""

import torch
import triton
import triton.language as tl


@triton.jit
def _select(
    Dots,
    Norm,
    CodeNorm,
    Book,
    Latents,
    Out,
    Indices,
    K: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    LB: tl.constexpr,
    LD: tl.constexpr,
    LT: tl.constexpr,
    STE: tl.constexpr,
    BLOCK: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    dots = tl.load(Dots + row * K + k, k < K, 0)
    norm = tl.load(Norm + row)
    cnorm = tl.load(CodeNorm + k, k < K, 0)
    # Preserve both FP32 roundings. Dropping the common row norm can change ties.
    distance = (norm - dots) + cnorm
    distance = tl.where(k < K, distance, float("inf"))
    # torch.max(-distance) chooses the first NaN, or first finite minimum.
    nan_index = tl.min(tl.where((distance != distance) & (k < K), k, 2147483647), 0)
    finite = tl.where(distance == distance, distance, float("inf"))
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
        raise ValueError("Expected B,D,T latents and K,D codebook")
    b, d, t = latents.shape
    k, book_d = codebook.shape
    if (
        d != book_d
        or not k
        or k > 65536
        or dots.shape != (b * t, k)
        or row_norm.shape != (b * t, 1)
        or codebook_norm.shape != (1, k)
    ):
        raise ValueError("Incompatible quantizer shapes or unsupported codebook size")
    tensors = (dots, row_norm, codebook_norm, codebook, latents)
    if any(
        x.dtype != torch.float32 or not x.is_cuda or x.device != dots.device
        for x in tensors
    ):
        raise ValueError("Quantizer fusion requires same-device CUDA FP32 inputs")
    if not all(x.is_contiguous() for x in tensors[:-1]):
        raise ValueError("Distance inputs and codebook must be contiguous")
    if straight_through and latents.stride() != (d * t, t, 1):
        values, indices = select(dots, row_norm, codebook_norm, codebook, latents)
        return latents + (values - latents), indices
    indices = torch.empty((b, t), device=dots.device, dtype=torch.long)
    output = (
        torch.empty((b, d, t), device=dots.device, dtype=torch.float32)
        if straight_through
        else torch.empty((b, t, d), device=dots.device, dtype=torch.float32)
    )
    if b * t:
        _select[(b * t,)](
            dots,
            row_norm,
            codebook_norm,
            codebook,
            latents,
            output,
            indices,
            k,
            d,
            t,
            *latents.stride(),
            straight_through,
            triton.next_power_of_2(k),
            triton.next_power_of_2(d),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (output if straight_through else output.transpose(1, 2)), indices

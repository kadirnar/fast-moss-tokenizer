"""Pinned FP32 SIMT FFN matrices, with the original 256-term partitions.

Internal entry point: MatrixRuntime validates operands, model and environment.
No Tensor Core instructions or reduced-precision operands are used.
"""
import torch
import triton
import triton.language as tl


SHAPES = {(24, 5120, 1280), (24, 1280, 5120)}


@triton.jit
def _partials(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(CHUNK, BK)):
        local = block * BK + kk
        k = part * CHUNK + local
        x = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K) & (local[None, :] < CHUNK), 0)
        w = tl.load(W + k[:, None] * N + n[None, :],
                    (n[None, :] < N) & (k[:, None] < K) & (local[:, None] < CHUNK), 0)
        acc = tl.dot(x, w, acc, input_precision='ieee')
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce(P, Y, NUMEL: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(P + i, i < NUMEL, 0)
    for part in range(1, PARTS):
        acc = acc + tl.load(P + part * NUMEL + i, i < NUMEL, 0)
    tl.store(Y + i, acc, i < NUMEL)


def linear(x, packed):
    """Validated contiguous X=(24,K), W=(K,N), with N/K in SHAPES."""
    k, n = packed.shape
    if (tuple(x.shape) != (24, k) or (24, n, k) not in SHAPES
            or not x.is_cuda or x.dtype != torch.float32 or packed.dtype != x.dtype
            or x.device != packed.device or not x.is_contiguous() or not packed.is_contiguous()):
        raise ValueError('Expected a supported contiguous CUDA FP32 ordered matrix shape')
    parts = k // 256
    partials = torch.empty((parts, 24, n), device=x.device, dtype=x.dtype)
    out = torch.empty((24, n), device=x.device, dtype=x.dtype)
    _partials[(1, n // 128, parts)](x, packed, partials, 24, n, k, 256, 32, 128, 32,
                                   num_warps=4, enable_fp_fusion=False)
    _reduce[(24 * n // 256,)](partials, out, 24*n, parts, 256, enable_fp_fusion=False)
    return out

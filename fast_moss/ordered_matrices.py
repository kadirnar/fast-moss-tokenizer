"""Pinned FP32 SIMT attention/FFN matrices, with measured native partitions.

Internal entry point: MatrixRuntime validates operands, model and environment.
No Tensor Core instructions or reduced-precision operands are used.
"""
import torch
import triton
import triton.language as tl


# (M,N,K): (partition terms, tile M, tile N, tile K, warps, stages).
# See ordered_shapes_confirm.json for the expanded empirical arithmetic gate.
CONFIGS = {
    (24, 768, 768): (96, 32, 128, 32, 4, 3),
    (24, 768, 3072): (96, 32, 128, 32, 4, 2),
    (24, 1280, 5120): (256, 32, 128, 32, 4, 3),
    (24, 2304, 768): (96, 32, 64, 32, 4, 3),
    (24, 3072, 768): (96, 32, 64, 32, 4, 2),
    (24, 3840, 1280): (96, 32, 128, 32, 4, 2),
    (24, 5120, 1280): (256, 32, 128, 32, 4, 3),
    (48, 768, 3072): (160, 32, 64, 32, 4, 3),
    (48, 2304, 768): (128, 32, 64, 32, 4, 2),
    (96, 768, 768): (128, 32, 128, 32, 4, 3),
    (96, 768, 3072): (288, 32, 128, 32, 4, 2),
    (96, 2304, 768): (128, 32, 64, 32, 4, 2),
    (96, 3072, 768): (192, 32, 128, 32, 4, 2),
    (192, 768, 768): (128, 32, 64, 32, 4, 2),
    (192, 768, 3072): (160, 64, 64, 32, 4, 3),
}
SHAPES = set(CONFIGS)


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


def linear(x, packed, *, stages=None):
    """Validated contiguous X=(M,K), W=(K,N), with shape-specific FMA partitions."""
    if x.ndim != 2 or packed.ndim != 2:
        raise ValueError('Expected a supported contiguous CUDA FP32 ordered matrix shape')
    m, k = x.shape
    n = packed.shape[1]
    shape = (m, n, k)
    if (packed.shape[0] != k or shape not in CONFIGS
            or not x.is_cuda or x.dtype != torch.float32 or packed.dtype != x.dtype
            or x.device != packed.device or not x.is_contiguous() or not packed.is_contiguous()):
        raise ValueError('Expected a supported contiguous CUDA FP32 ordered matrix shape')
    chunk, bm, bn, bk, warps, configured_stages = CONFIGS[shape]
    parts = triton.cdiv(k, chunk)
    partials = torch.empty((parts, m, n), device=x.device, dtype=x.dtype)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    _partials[(triton.cdiv(m, bm), triton.cdiv(n, bn), parts)](
        x, packed, partials, m, n, k, chunk, bm, bn, bk, num_warps=warps,
        num_stages=configured_stages if stages is None else stages, enable_fp_fusion=False)
    _reduce[(triton.cdiv(m*n, 256),)](partials, out, m*n, parts, 256, enable_fp_fusion=False)
    return out

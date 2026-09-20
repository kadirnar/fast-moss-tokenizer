"""Pinned FP32 SIMT attention/FFN matrices, with measured native partitions.

Internal entry point: MatrixRuntime validates operands, model and environment.
No Tensor Core instructions or reduced-precision operands are used.
"""
import torch
import triton
import triton.language as tl


# (M,N,K): (partition terms, tile M, tile N, tile K, warps, stages).
# See ordered_shapes_confirm.json and ordered_remaining_confirm.json for
# empirical arithmetic gates on learned weights and stress inputs.
CONFIGS = {
    (24, 768, 768): (96, 32, 128, 32, 4, 3),
    (24, 768, 3072): (96, 32, 128, 32, 4, 2),
    (24, 1280, 1280): (96, 32, 64, 32, 4, 3),
    (24, 1280, 5120): (256, 32, 128, 32, 4, 3),
    (24, 2304, 768): (96, 32, 64, 32, 4, 3),
    (24, 3072, 768): (96, 32, 64, 32, 4, 2),
    (24, 3840, 1280): (96, 32, 128, 32, 4, 2),
    (24, 5120, 1280): (256, 32, 128, 32, 4, 3),
    (32, 384, 768): (64, 16, 64, 32, 4, 2),
    (32, 768, 384): (64, 16, 64, 32, 4, 2),
    (32, 768, 768): (96, 32, 128, 32, 4, 3),
    (32, 768, 3072): (96, 32, 128, 32, 4, 2),
    (32, 2304, 768): (96, 32, 64, 32, 4, 3),
    (32, 3072, 768): (512, 32, 128, 32, 4, 3),
    (48, 768, 768): (96, 32, 64, 32, 4, 3),
    (48, 768, 3072): (160, 32, 64, 32, 4, 3),
    (48, 2304, 768): (128, 32, 64, 32, 4, 2),
    (48, 3072, 768): (96, 64, 64, 32, 4, 2),
    (64, 240, 768): (64, 64, 64, 32, 4, 3),
    (64, 384, 768): (96, 64, 64, 32, 4, 3),
    (64, 768, 240): (80, 16, 64, 32, 4, 2),
    (64, 768, 384): (96, 64, 64, 32, 4, 3),
    (64, 768, 768): (96, 32, 64, 32, 4, 3),
    (64, 768, 3072): (160, 32, 64, 32, 4, 3),
    (64, 2304, 768): (128, 32, 64, 32, 4, 2),
    (64, 3072, 768): (96, 64, 64, 32, 4, 2),
    (96, 768, 768): (128, 32, 128, 32, 4, 3),
    (96, 768, 3072): (288, 32, 128, 32, 4, 2),
    (96, 2304, 768): (128, 32, 64, 32, 4, 2),
    (96, 3072, 768): (192, 32, 128, 32, 4, 2),
    (192, 768, 768): (128, 32, 64, 32, 4, 2),
    (192, 768, 3072): (160, 64, 64, 32, 4, 3),
    (192, 2304, 768): (160, 64, 128, 32, 4, 3),
    (192, 3072, 768): (192, 64, 64, 32, 4, 2),
}
SHAPES = set(CONFIGS)

# These native paths preserve -0; the other validated paths add +0 in their
# epilogue. Tail blocks must not manufacture extra FMAs for preserved -0.
PRESERVE_SIGNED_ZERO = {
    (24,5120,1280), (96,3072,768), (192,768,3072),
    (192,2304,768), (192,3072,768),
}


@triton.jit
def _partials(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              EXACT_TAIL: tl.constexpr = False):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    blocks = tl.cdiv(CHUNK, BK)
    if EXACT_TAIL:
        blocks = tl.cdiv(tl.minimum(CHUNK, K - part * CHUNK), BK)
    for block in range(blocks):
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
def _reduce(P, Y, NUMEL: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr,
            ADD_ZERO: tl.constexpr = False):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(P + i, i < NUMEL, 0)
    for part in range(1, PARTS):
        acc = acc + tl.load(P + part * NUMEL + i, i < NUMEL, 0)
    if ADD_ZERO:
        acc = tl.inline_asm_elementwise('add.rn.f32 $0, $1, 0f00000000;',
            constraints='=f,f', args=[acc], dtype=tl.float32, is_pure=True, pack=1)
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
        num_stages=configured_stages if stages is None else stages, enable_fp_fusion=False,
        EXACT_TAIL=shape in PRESERVE_SIGNED_ZERO and k % chunk != 0)
    _reduce[(triton.cdiv(m*n, 256),)](partials, out, m*n, parts, 256, enable_fp_fusion=False,
        ADD_ZERO=shape not in PRESERVE_SIGNED_ZERO)
    return out

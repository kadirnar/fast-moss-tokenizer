"""V2 mixed-precision pointwise fusion with separate native FP32 roundings."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    N: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    K3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = i % (D // 2)
    t = i // (D // 2) % T
    h = i // (D // 2 * T) % H
    b = i // (D // 2 * T * H)
    qi = b * Q0 + h * Q1 + t * Q2 + 2 * pair * Q3
    ki = b * K0 + h * K1 + t * K2 + 2 * pair * K3
    phase = (b * T + t) * (D // 2) + pair
    qr = tl.load(Q + qi, i < N, 0).to(tl.float32)
    qj = tl.load(Q + qi + Q3, i < N, 0).to(tl.float32)
    kr = tl.load(K + ki, i < N, 0).to(tl.float32)
    kj = tl.load(K + ki + K3, i < N, 0).to(tl.float32)
    c = tl.load(C + phase, i < N, 0)
    s = tl.load(S + phase, i < N, 0)
    tl.store(OQ + 2 * i, qr * c - qj * s, i < N)
    tl.store(OQ + 2 * i + 1, qr * s + qj * c, i < N)
    tl.store(OK + 2 * i, kr * c - kj * s, i < N)
    tl.store(OK + 2 * i + 1, kr * s + kj * c, i < N)


def rotate(q, k, cos, sin):
    if q.ndim != 4 or q.shape != k.shape or q.shape[-1] < 2 or q.shape[-1] % 2:
        raise ValueError("Expected equal B,H,T,even-D rotary inputs")
    b, h, t, d = q.shape
    if (
        q.dtype not in (torch.bfloat16, torch.float32)
        or k.dtype != q.dtype
        or cos.dtype != torch.float32
        or sin.dtype != torch.float32
        or not q.is_cuda
        or any(x.device != q.device for x in (k, cos, sin))
        or cos.shape != (b, t, d // 2)
        or sin.shape != cos.shape
        or not cos.is_contiguous()
        or not sin.is_contiguous()
    ):
        raise ValueError(
            "Expected same-device BF16/FP32 Q/K and contiguous FP32 rotary tables"
        )
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    n = q.numel() // 2
    if n:
        _rotate[(triton.cdiv(n, 256),)](
            q,
            k,
            cos,
            sin,
            oq,
            ok,
            n,
            h,
            t,
            d,
            *q.stride(),
            *k.stride(),
            256,
            enable_fp_fusion=False,
        )
    return oq, ok


@triton.jit
def _scale_add(X, U, S, Y, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    u = tl.load(U + i, i < N, 0).to(tl.float32)
    s = tl.load(S + i % C)
    tl.store(Y + i, x + u * s, i < N)


def scale_add(x, update, scale):
    if (
        x.shape != update.shape
        or scale.shape != (x.shape[-1],)
        or x.dtype not in (torch.bfloat16, torch.float32)
        or update.dtype not in (torch.bfloat16, torch.float32)
        or scale.dtype != torch.float32
        or not x.is_cuda
        or update.device != x.device
        or scale.device != x.device
    ):
        raise ValueError(
            "Expected equal BF16/FP32 residuals and same-device FP32 channel scale"
        )
    if not all(t.is_contiguous() for t in (x, update, scale)):
        return x.float() + scale * update
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    if x.numel():
        _scale_add[(triton.cdiv(x.numel(), 256),)](
            x, update, scale, out, x.numel(), x.shape[-1], 256, enable_fp_fusion=False
        )
    return out

"""Prepare heterogeneous stream tails without reading device state on the host."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(X, LENGTHS, REQUESTED, CLOSED, ACTIVE, Y, EFFECTIVE,
             B: tl.constexpr, Q: tl.constexpr, T: tl.constexpr,
             X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
             ENCODE: tl.constexpr, RATE: tl.constexpr, HAS_REQUESTED: tl.constexpr,
             REQUESTED_STRIDE: tl.constexpr,
             BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(LENGTHS + b)
    active = (n > 0) & ~tl.load(CLOSED + b)
    if HAS_REQUESTED:
        active = active & tl.load(REQUESTED + b * REQUESTED_STRIDE)
    t = i % T
    q = i // T
    if ENCODE:
        src = b * X0 + t * X2
        dst = b * T + t
        effective = tl.cdiv(n, RATE) * RATE
    else:
        src = q * X0 + b * X1 + t * X2
        dst = (q * B + b) * T + t
        effective = n
    value = tl.load(X + src, (i < Q * T) & (t < n) & active, 0)
    tl.store(Y + dst, value, i < Q * T)
    if tl.program_id(1) == 0:
        tl.store(ACTIVE + b, active)
        tl.store(EFFECTIVE + b, tl.where(active, effective, 0))


def prepare(chunk, raw_lengths, requested, closed, active, *, encode, width, rate):
    """Internal API: StreamingSession validates shapes, lengths, and device."""
    b = closed.numel()
    q = 1 if encode else chunk.shape[0]
    shape = (b, 1, width) if encode else (q, b, width)
    output = torch.empty(shape, device=chunk.device, dtype=chunk.dtype)
    effective = torch.empty_like(raw_lengths)
    _prepare[(b, triton.cdiv(q * width, 256))](
        chunk, raw_lengths, closed if requested is None else requested, closed, active,
        output, effective, b, q, width, *chunk.stride(), encode, rate,
        requested is not None, requested.stride(0) if requested is not None else 0, 256)
    return output, effective

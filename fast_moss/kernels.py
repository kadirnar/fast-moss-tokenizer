"""FP32 kernels preserving separate multiply/add rounding of the reference."""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_add(X, U, S, Y, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    u = tl.load(U + i, i < N, 0)
    s = tl.load(S + i % C)
    y = x + u * s
    tl.store(Y + i, y, i < N)


def scale_add(x, update, scale):
    if x.dtype != torch.float32 or update.dtype != x.dtype or scale.dtype != x.dtype:
        raise TypeError("The exact residual kernel currently supports FP32 only")
    if x.shape != update.shape or scale.shape != (x.shape[-1],):
        raise ValueError("Expected equal residual/update shapes and one scale per channel")
    if not (x.is_cuda and x.device == update.device == scale.device):
        raise ValueError("Inputs must be on the same CUDA device")
    if not all(t.is_contiguous() for t in (x, update, scale)):
        return x + update * scale
    out = torch.empty_like(x)
    if x.numel():
        _scale_add[(triton.cdiv(x.numel(), 256),)](
            x, update, scale, out, x.numel(), x.shape[-1], 256, enable_fp_fusion=False)
    return out

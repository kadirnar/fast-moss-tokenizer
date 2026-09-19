"""CuTe DSL residual fusion with explicitly rounded CUDA PTX operations."""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cuda.bindings import driver as cuda


@dsl_user_op
def _rounded_scale_add(x, u, s, *, loc=None, ip=None):
    result = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [x.ir_value(loc=loc, ip=ip), u.ir_value(loc=loc, ip=ip), s.ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 p; mul.rn.f32 p, $2, $3; add.rn.f32 $0, $1, p; }",
        "=f,f,f,f", has_side_effects=False, asm_dialect=0, loc=loc, ip=ip,
    )
    return cutlass.Float32(result)


@cute.kernel
def _kernel(x: cute.Tensor, u: cute.Tensor, s: cute.Tensor, y: cute.Tensor,
            n: cutlass.Constexpr, channels: cutlass.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + thread
    if i < n:
        y[i] = _rounded_scale_add(x[i], u[i], s[i % channels])


@cute.jit
def _launch(x: cute.Tensor, u: cute.Tensor, s: cute.Tensor, y: cute.Tensor,
            n: cutlass.Constexpr, channels: cutlass.Constexpr, stream: cuda.CUstream):
    _kernel(x, u, s, y, n, channels).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


_compiled = {}


def scale_add(x, update, scale):
    if any(t.dtype != torch.float32 for t in (x, update, scale)):
        raise TypeError("The exact CuTe residual kernel supports FP32 only")
    if x.shape != update.shape or scale.shape != (x.shape[-1],):
        raise ValueError("Expected equal residual/update shapes and one scale per channel")
    if not (x.is_cuda and x.device == update.device == scale.device):
        raise ValueError("Inputs must be on the same CUDA device")
    if not all(t.is_contiguous() for t in (x, update, scale)):
        return x + update * scale
    out = torch.empty_like(x)
    if not x.numel():
        return out
    args = tuple(from_dlpack(t.view(-1)) for t in (x, update, scale, out))
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.device, x.numel(), x.shape[-1])
    if key not in _compiled:
        _compiled[key] = cute.compile(_launch, *args, x.numel(), x.shape[-1], stream)
    _compiled[key](*args, stream)
    return out

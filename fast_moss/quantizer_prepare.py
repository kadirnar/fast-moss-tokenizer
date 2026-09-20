"""Pinned eight-channel LFQ normalization and distance preparation.

Mirrors PyTorch 2.8 Reduce.cuh: adjacent shuffle tree for contiguous
eight-element rows, four cyclic accumulators for transposed single-batch rows.
Used by the v2 optimization context with Triton LFQ selection.
"""

import ctypes
import hashlib

import torch

from .cuda import compiler, _check


SOURCE = r"""
__device__ __forceinline__ float sum8(const float* v, bool strided) {
  if(strided) {
    float a=__fadd_rn(v[0],v[4]), b=__fadd_rn(v[1],v[5]);
    float c=__fadd_rn(v[2],v[6]), d=__fadd_rn(v[3],v[7]);
    return __fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);
  }
  return __fadd_rn(__fadd_rn(__fadd_rn(v[0],v[1]),__fadd_rn(v[2],v[3])),
                   __fadd_rn(__fadd_rn(v[4],v[5]),__fadd_rn(v[6],v[7])));
}
extern "C" __global__ void quantizer_prepare(const float* X, float* E,
    float* Norm, float* Twice, float* Denom, int batch, int time) {
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=batch*time)return;
  bool strided=batch==1 && time>1;
  float x[8],square[8],e[8],esquare[8];
  #pragma unroll
  for(int d=0;d<8;++d) {
    x[d]=X[(row/time)*8*time+d*time+row%time];
    square[d]=__fmul_rn(x[d],x[d]);
  }
  float total;
  if(strided) {
    float a=__fmaf_rn(x[4],x[4],square[0]);
    float b=__fmaf_rn(x[5],x[5],square[1]);
    float c=__fmaf_rn(x[6],x[6],square[2]);
    float d=__fmaf_rn(x[7],x[7],square[3]);
    total=__fadd_rn(__fadd_rn(__fadd_rn(a,b),c),d);
  } else total=sum8(square,false);
  float denom=__fsqrt_rn(total);
  // clamp_min propagates NaN, unlike fmaxf.
  denom=denom<1.e-12f ? 1.e-12f : denom;
  #pragma unroll
  for(int d=0;d<8;++d) {
    e[d]=__fdiv_rn(x[d],denom);
    esquare[d]=__fmul_rn(e[d],e[d]);
    int out=strided ? d*time+row : row*8+d;
    E[out]=e[d];Twice[out]=__fmul_rn(2.f,e[d]);
  }
  Norm[row]=sum8(esquare,strided);
  Denom[row]=denom;
}
"""
CACHE = {}


def compile_kernel():
    device = torch.cuda.current_device()
    if device in CACHE:
        return CACHE[device]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm quantizer preparation before capture")
    cu, nvrtc = compiler()
    program = _check(
        nvrtc.nvrtcCreateProgram(SOURCE.encode(), b"quantizer_prepare.cu", 0, [], [])
    )
    try:
        options = [
            b"--gpu-architecture=sm_120",
            b"--std=c++17",
            b"--ftz=false",
            b"--fmad=false",
        ]
        result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(result[0]):
            log = b" " * _check(nvrtc.nvrtcGetProgramLogSize(program))
            _check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode())
        blob = b" " * _check(nvrtc.nvrtcGetCUBINSize(program))
        _check(nvrtc.nvrtcGetCUBIN(program, blob))
    finally:
        _check(nvrtc.nvrtcDestroyProgram(program))
    module = _check(cu.cuModuleLoadData(blob))
    try:
        fn = _check(cu.cuModuleGetFunction(module, b"quantizer_prepare"))
    except BaseException:
        _check(cu.cuModuleUnload(module))
        raise
    attr = cu.CUfunction_attribute
    resources = {
        name: _check(cu.cuFuncGetAttribute(a, fn))
        for name, a in [
            ("registers", attr.CU_FUNC_ATTRIBUTE_NUM_REGS),
            ("local_bytes", attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
            ("shared_bytes", attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        ]
    }
    resources.update(
        source_sha256=hashlib.sha256(SOURCE.encode()).hexdigest(),
        cubin_sha256=hashlib.sha256(blob).hexdigest(),
    )
    CACHE[device] = module, fn, resources
    return CACHE[device]


def prepare(x):
    if (
        x.ndim != 3
        or x.shape[1] != 8
        or not x.is_cuda
        or x.dtype != torch.float32
        or not x.is_contiguous()
        or x.shape[0] < 1
        or x.shape[2] < 1
        or x.numel() > 2147483647
    ):
        raise ValueError("Expected nonempty contiguous CUDA FP32 B,8,T latents")
    b, _, t = x.shape
    with torch.cuda.device(x.device):
        cu, _ = compiler()
        _, fn, _ = compile_kernel()
        normalized = torch.empty(
            (8, t) if b == 1 and t > 1 else (b * t, 8), device=x.device, dtype=x.dtype
        )
        if b == 1 and t > 1:
            normalized = normalized.T
        twice = torch.empty_strided(
            normalized.shape, normalized.stride(), device=x.device, dtype=x.dtype
        )
        norm = torch.empty((b * t, 1), device=x.device, dtype=x.dtype)
        denom = torch.empty_like(norm)
        values = [
            ctypes.c_void_p(v.data_ptr()) for v in (x, normalized, norm, twice, denom)
        ]
        values += [ctypes.c_int(b), ctypes.c_int(t)]
        ptrs = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
        _check(
            cu.cuLaunchKernel(
                fn,
                (b * t + 127) // 128,
                1,
                1,
                128,
                1,
                1,
                0,
                cu.CUstream(torch.cuda.current_stream().cuda_stream),
                ctypes.addressof(ptrs),
                0,
            )
        )
    return normalized, norm, twice, denom

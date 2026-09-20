"""Pinned single-block FP32 K partitions with the native tile/lane order.

No weight packing or global partial buffer. Explicit 16-lane shuffle membership
is valid for full warps and the final half warp of a 48-thread launch.
"""
import ctypes
import torch
from importlib.metadata import PackageNotFoundError
from .normalization import _check, compiler as _compiler

CONFIGS = {
    (4, 3072, 768): (4, 8, 4, False),
    (4, 768, 768): (4, 4, 16, True),
    (6, 3072, 768): (6, 8, 4, False),
    (6, 2304, 768): (6, 2, 4, True),
    (12, 768, 768): (6, 8, 16, False),
}

SOURCE = r'''
__device__ __forceinline__ float add(float a,float b){
    float value;asm("add.rn.f32 %0, %1, %2;":"=f"(value):"f"(a),"f"(b));return value;
}
extern "C" __global__ void cta_tiled(const float* X,const float* W,float* Y){
    constexpr int PARTS=K/256;
    int lane=threadIdx.x%16;
    int output=(threadIdx.x/16)%COLS;
    int part=threadIdx.x/(16*COLS);
    int n=blockIdx.x*COLS+output,m=blockIdx.y*ROWS;
    float acc[ROWS]={};
    #pragma unroll UNROLL
    for(int step=0;step<16;++step){
        int k=part*256+step*16+lane;
        float w=n<N?W[n*K+k]:0.f;
        #pragma unroll
        for(int r=0;r<ROWS;++r){
            float x=m+r<M?X[(m+r)*K+k]:0.f;
            acc[r]=__fmaf_rn(x,w,acc[r]);
        }
    }
    __shared__ float partial[PARTS*ROWS*COLS*16];
    #pragma unroll
    for(int r=0;r<ROWS;++r)partial[((part*ROWS+r)*COLS+output)*16+lane]=acc[r];
    __syncthreads();
    // Every output group has exactly 16 participating lanes, including
    // the half-warp at the end of a 48-thread block. Do not sample activemask
    // inside the divergent reducer branch under independent scheduling.
    unsigned mask=0xffffu << (threadIdx.x%32/16*16);
    #pragma unroll
    for(int r=0;r<ROWS;++r){
        if(part==(DISTRIBUTE ? r%PARTS : 0)){
            float sum=0.f;
            #pragma unroll
            for(int p=0;p<PARTS;++p)sum=add(sum,partial[((p*ROWS+r)*COLS+output)*16+lane]);
            float value=__shfl_sync(mask,sum,0,16);
            #pragma unroll
            for(int l=1;l<16;++l)value=add(value,__shfl_sync(mask,sum,l,16));
            if(lane==0 && m+r<M && n<N)Y[(m+r)*N+n]=value;
        }
    }
}
'''

_KERNELS = {}


def compiler():
    try:
        return _compiler()
    except (ValueError, PackageNotFoundError) as error:
        raise ValueError('CUDA matrices require the pinned cuda extra') from error


def _compile(shape, config, bindings):
    key = (torch.cuda.current_device(), shape, config)
    if key in _KERNELS:
        return _KERNELS[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm CUDA matrices before graph capture')
    cu, nvrtc = bindings
    m, n, k = shape
    rows, cols, unroll, distribute = config
    definitions = dict(M=m, N=n, K=k, ROWS=rows, COLS=cols, UNROLL=unroll, DISTRIBUTE=int(distribute))
    source = ('\n'.join(f'#define {a} {b}' for a, b in definitions.items()) + '\n' + SOURCE).encode()
    program = _check(nvrtc.nvrtcCreateProgram(source, b'cta_tiled.cu', 0, [], []))
    try:
        options = [b'--gpu-architecture=sm_120', b'--std=c++17', b'--ftz=false', b'--fmad=false']
        result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if int(result[0]):
            log = b' ' * _check(nvrtc.nvrtcGetProgramLogSize(program))
            _check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode())
        blob = b' ' * _check(nvrtc.nvrtcGetCUBINSize(program))
        _check(nvrtc.nvrtcGetCUBIN(program, blob))
    finally:
        _check(nvrtc.nvrtcDestroyProgram(program))
    module = _check(cu.cuModuleLoadData(blob))
    try:
        function = _check(cu.cuModuleGetFunction(module, b'cta_tiled'))
    except BaseException:
        _check(cu.cuModuleUnload(module))
        raise
    # Process lifetime keeps compiled kernels valid for captured graphs.
    _KERNELS[key] = module, function
    return module, function


def linear(x, weight, bindings):
    if (x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]
            or not x.is_cuda or x.dtype != torch.float32 or weight.dtype != x.dtype
            or weight.device != x.device or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('Expected supported contiguous CUDA FP32 matrix operands')
    m, k = x.shape
    n = weight.shape[0]
    config = CONFIGS.get((m, n, k))
    if config is None:
        raise ValueError('Expected supported contiguous CUDA FP32 matrix operands')
    rows, cols, _, _ = config
    with torch.cuda.device(x.device):
        cu, _ = bindings
        _, function = _compile((m, n, k), config, bindings)
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        values = [ctypes.c_void_p(v.data_ptr()) for v in (x, weight, out)]
        pointers = (ctypes.c_void_p * 3)(*(ctypes.addressof(v) for v in values))
        _check(cu.cuLaunchKernel(function, (n+cols-1)//cols, (m+rows-1)//rows, 1,
            16*cols*(k//256), 1, 1, 0, cu.CUstream(torch.cuda.current_stream().cuda_stream),
            ctypes.addressof(pointers), 0))
    return out

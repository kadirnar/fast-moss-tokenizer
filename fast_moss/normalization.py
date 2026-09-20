"""Pinned CUDA FP32 LayerNorm with the native PyTorch Welford reduction tree.

Arithmetic reference: PyTorch v2.8.0 aten/native/cuda/layer_norm_kernel.cu.
Fixed counts preserve the native four-value logical thread order. Explicit
mean FMA orientation is required even when the merge coefficients are equal.
"""
import ctypes
from importlib.metadata import version
from threading import get_ident
from types import MethodType

import torch

from . import storage_epoch

CONFIGS = {
    (1, 1280): (False, 128, 1),
    (2, 768): (False, 128, 1),
    (3, 1280): (False, 128, 1),
    (4, 768): (False, 128, 1),
    (6, 768): (False, 128, 1),
    (8, 768): (False, 128, 1),
    (8, 1280): (False, 128, 1),
    (12, 768): (False, 128, 1),
    (16, 768): (False, 128, 1),
    (24, 768): (False, 128, 1),
    (32, 768): (False, 128, 1),
    (40, 1280): (False, 128, 1),
    (64, 768): (False, 128, 1),
    (80, 768): (False, 128, 1),
    (128, 1280): (False, 128, 1),
    (160, 768): (False, 128, 1),
    (256, 768): (False, 128, 1),
    (320, 768): (False, 128, 1),
    (512, 768): (False, 128, 4),
    (1024, 768): (True, 32, 4),
}

SOURCE = r'''
struct Stats {float mean,var,count;};
__device__ __forceinline__ float fixed_mean(float current,float remote,float ra,float rb){
    return __fmaf_rn(ra,remote,__fmul_rn(rb,current));
}
template<int A,int B> __device__ __forceinline__ Stats fixed_merge(Stats current,Stats remote){
    constexpr float inv=1.f/float(A+B),ra=float(B)*inv,rb=float(A)*inv;
    float d=current.mean-remote.mean;
    return {fixed_mean(current.mean,remote.mean,ra,rb),
            remote.var+current.var+d*d*float(B)*rb,float(A+B)};
}
template<int COUNT> __device__ __forceinline__ Stats fixed_stats(const float* x,int logical){
    Stats s={};
    #pragma unroll
    for(int step=0;step<COUNT/4;++step){
        float4 v=reinterpret_cast<const float4*>(x)[logical+step*128];
        float values[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int j=0;j<4;++j){
            float d=values[j]-s.mean;
            float m=s.mean+d*(1.f/float(step*4+j+1));
            s.var=s.var+d*(values[j]-m);s.mean=m;
        }
    }
    #pragma unroll
    for(int i=0;i<5;++i){
        int offset=16>>i;
        Stats other={__shfl_down_sync(0xffffffff,s.mean,offset),
                     __shfl_down_sync(0xffffffff,s.var,offset),0.f};
        // Counts double at each native shuffle level. Compiler unrolling makes
        // each coefficient constant without reordering the floating operations.
        float count=float(COUNT*(1<<i));
        float d=s.mean-other.mean;
        s={fixed_mean(s.mean,other.mean,.5f,.5f),other.var+s.var+d*d*count*.5f,count*2.f};
    }
    return s;
}
extern "C" __global__ void layer_norm(const float* __restrict__ X,
    const float* G,const float* B,float* Y,float* Mean,float* Rstd,float eps,int rows){
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int row=REGISTER ? blockIdx.x*(THREADS/32)+warp : blockIdx.x;
    if(row>=rows)return;
    constexpr int SLOTS=REGISTER?4:1;
    Stats st[SLOTS]={};
    #pragma unroll
    for(int slot=0;slot<SLOTS;++slot){
        int logical=lane+32*(REGISTER?slot:warp);
        if((REGISTER?slot:warp)<2)st[slot]=fixed_stats<((N/4+127)/128)*4>(X+row*N,logical);
        else st[slot]=fixed_stats<(N/4/128)*4>(X+row*N,logical);
    }
    float mean,variance;
    if(REGISTER){
        if(lane==0){
            constexpr int H=((N/4+127)/128)*128, L=(N/4/128)*128;
            st[0]=fixed_merge<H,L>(st[0],st[2]);st[1]=fixed_merge<H,L>(st[1],st[3]);
            st[0]=fixed_merge<H+L,H+L>(st[0],st[1]);
        }
        mean=__shfl_sync(0xffffffff,st[0].mean,0);
        variance=__fdiv_rn(__shfl_sync(0xffffffff,st[0].var,0),float(N));
    }else{
        __shared__ Stats shared[4];
        if(lane==0)shared[warp]=st[0];
        __syncthreads();
        if(threadIdx.x==0){
            constexpr int H=((N/4+127)/128)*128, L=(N/4/128)*128;
            Stats a=fixed_merge<H,L>(shared[0],shared[2]),b=fixed_merge<H,L>(shared[1],shared[3]);
            shared[0]=fixed_merge<H+L,H+L>(a,b);
        }
        __syncthreads();
        mean=shared[0].mean;variance=__fdiv_rn(shared[0].var,float(N));
    }
    float rs=rsqrtf(variance+eps);
    int start=REGISTER?lane:threadIdx.x,stride=REGISTER?32:128;
    #pragma unroll UNROLL
    for(int idx=start;idx<N/4;idx+=stride){
        float4 v=reinterpret_cast<const float4*>(X+row*N)[idx];
        float4 g=reinterpret_cast<const float4*>(G)[idx],b=reinterpret_cast<const float4*>(B)[idx],out;
        out.x=g.x*(rs*(v.x-mean))+b.x;out.y=g.y*(rs*(v.y-mean))+b.y;
        out.z=g.z*(rs*(v.z-mean))+b.z;out.w=g.w*(rs*(v.w-mean))+b.w;
        reinterpret_cast<float4*>(Y+row*N)[idx]=out;
    }
    if((REGISTER&&lane==0)||(!REGISTER&&threadIdx.x==0)){Mean[row]=mean;Rstd[row]=rs;}
}
'''

# Modules remain loaded for process lifetime, including captured raw kernels.
_KERNELS = {}
_NATIVE_FORWARD = torch.nn.LayerNorm.forward


def _check(result):
    if int(result[0]):
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def compiler():
    if (version('cuda-bindings') != '13.4.2'
            or version('nvidia-cuda-nvrtc-cu12') != '12.8.93'):
        raise ValueError('CUDA LayerNorm requires the validated normalization extra')
    from cuda.bindings import driver, nvrtc
    if tuple(_check(nvrtc.nvrtcVersion())) != (12, 8):
        raise ValueError('CUDA LayerNorm requires NVRTC 12.8')
    return driver, nvrtc


def _compile(device, n, config, bindings):
    key = (device, n, config)
    if key in _KERNELS:
        return _KERNELS[key]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm CUDA LayerNorm before graph capture')
    cu, nvrtc = bindings
    register, threads, unroll = config
    definitions = dict(N=n, REGISTER=int(register), THREADS=threads, UNROLL=unroll)
    source = ('\n'.join(f'#define {k} {v}' for k, v in definitions.items()) + '\n' + SOURCE).encode()
    program = _check(nvrtc.nvrtcCreateProgram(source, b'layer_norm.cu', 0, [], []))
    try:
        options = [b'--gpu-architecture=sm_120', b'--std=c++17', b'--ftz=false', b'--fmad=true']
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
        function = _check(cu.cuModuleGetFunction(module, b'layer_norm'))
    except BaseException:
        _check(cu.cuModuleUnload(module))
        raise
    _KERNELS[key] = (module, function)
    return module, function


def owned_forward(module):
    """Allow FFN fusion only through this runtime's unchanged active wrapper."""
    runtime = getattr(module, '_fast_norm_runtime', None)
    return (isinstance(runtime, NormalizationRuntime) and runtime.active
            and runtime.forwards.get(module) is module.forward)


class NormalizationRuntime:
    """Reversible, single-host-thread owner of profiled frozen LayerNorm calls."""

    def __init__(self, model):
        from .matrices import load_profile
        load_profile(model)
        self.bindings = compiler()
        self.model = model
        self.device = next(model.parameters()).device
        self.thread = get_ident()
        self.active = self.used = False
        self.forwards = {}
        self.warmed = set()
        self.calls = 0

    def __enter__(self):
        if self.used or getattr(self.model, '_fast_norm_runtime', None) is not None:
            raise RuntimeError('Normalization runtime is already active or has been used')
        if getattr(self.model, '_fast_streaming_owner', None) is not None:
            raise RuntimeError('Enter normalization before opening a streaming session')
        if (self.model.training or any(p.requires_grad or p.dtype != torch.float32
                or p.device != self.device for p in self.model.parameters()) or self.device.type != 'cuda'):
            raise ValueError('CUDA LayerNorm requires a frozen FP32 model on one CUDA device')
        if get_ident() != self.thread:
            raise RuntimeError('Use normalization on its construction host thread')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Enter normalization before CUDA graph capture')
            torch.cuda.synchronize(self.device)
            storage_epoch.advance(self.device, context=True)
        self.used = self.active = True
        self.model._fast_norm_runtime = self
        try:
            for module in self.model.modules():
                if (type(module) is torch.nn.LayerNorm and 'forward' not in module.__dict__
                        and not hasattr(module, '_fast_norm_runtime')
                        and torch.nn.LayerNorm.forward is _NATIVE_FORWARD):
                    forward = MethodType(self._wrap(module.forward), module)
                    self.forwards[module] = forward
                    module._fast_norm_runtime = self
                    module.forward = forward
            return self
        except BaseException:
            self.close()
            raise

    def _wrap(self, original):
        def forward(module, x):
            if not self.active or get_ident() != self.thread:
                raise RuntimeError('Normalization call outside its active owner thread')
            g, b = module.weight, module.bias
            if (len(module.normalized_shape) != 1 or x.ndim < 1
                    or x.device != self.device or x.dtype != torch.float32
                    or not x.is_contiguous() or module.eps != 1e-5
                    or g is None or b is None or torch.is_autocast_enabled('cuda')
                    or (torch.is_grad_enabled() and any(t.requires_grad for t in (x, g, b)))):
                return original(x)
            n = module.normalized_shape[0]
            shape = (x.numel() // n, n) if n else (0, 0)
            if (x.shape[-1] != n or shape not in CONFIGS or g.shape != (n,) or b.shape != (n,)
                    or any(t.device != self.device or t.dtype != torch.float32
                           or not t.is_contiguous() or t.data_ptr() % 16 for t in (x, g, b))):
                return original(x)
            config = CONFIGS[shape]
            cu, _ = self.bindings
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device).cuda_stream
                key = (stream, shape)
                if torch.cuda.is_current_stream_capturing() and key not in self.warmed:
                    raise RuntimeError('Warm CUDA LayerNorm shape on this stream before graph capture')
                _, function = _compile(self.device, n, config, self.bindings)
                y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
                mean = torch.empty(shape[0], device=x.device, dtype=x.dtype)
                rs = torch.empty_like(mean)
                values = [ctypes.c_void_p(t.data_ptr()) for t in (x, g, b, y, mean, rs)]
                values += [ctypes.c_float(module.eps), ctypes.c_int(shape[0])]
                pointers = (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))
                register, threads, _ = config
                grid = (shape[0] + threads // 32 - 1) // (threads // 32) if register else shape[0]
                _check(cu.cuLaunchKernel(function, grid, 1, 1, threads, 1, 1, 0,
                    cu.CUstream(stream), ctypes.addressof(pointers), 0))
                self.warmed.add(key)
                self.calls += 1
                return y
        return forward

    def close(self):
        if not self.active:
            return
        try:
            torch.cuda.synchronize(self.device)
        finally:
            storage_epoch.advance(self.device, context=True)
            for module in self.forwards:
                module.__dict__.pop('forward', None)
                module.__dict__.pop('_fast_norm_runtime', None)
            self.forwards.clear()
            del self.model._fast_norm_runtime
            self.active = False

    def __exit__(self, *exc):
        self.close()

"""Exact ordered FFN epilogues using the pinned CUDA 12.8 math library."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn.modules.module as module_hooks
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .ordered_matrices import _partials


LIBDEVICE_SHA256 = '25498d6e3c64782833ff15a5f56665ff09e5d21177163148a1cf008c61733449'


def math_library():
    try:
        import nvidia.cuda_nvcc
        installed = version('nvidia-cuda-nvcc-cu12')
        path = Path(nvidia.cuda_nvcc.__path__[0]) / 'nvvm/libdevice/libdevice.10.bc'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (ImportError, PackageNotFoundError, OSError) as error:
        raise ValueError('FFN fusion requires the ffn extra: nvidia-cuda-nvcc-cu12==12.8.93') from error
    if installed != '12.8.93' or digest != LIBDEVICE_SHA256:
        raise ValueError('FFN fusion requires the validated CUDA 12.8.93 libdevice contents')
    return str(path)


@triton.jit
def _mul(x, y):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[x,y],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add(x, y):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;', constraints='=f,f,f', args=[x,y],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _ffn_reduce(P, X, S, Y, NUMEL:tl.constexpr, N:tl.constexpr, PARTS:tl.constexpr,
                 MODE:tl.constexpr, BLOCK:tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    acc = tl.load(P+i,i<NUMEL,0)
    for part in range(1,PARTS):
        acc = _add(acc,tl.load(P+part*NUMEL+i,i<NUMEL,0))
    if MODE == 'gelu':
        half = _mul(acc,0.5)
        erf = libdevice.erf(_mul(acc,0.7071067811865476))
        acc = _mul(half,_add(erf,1.0))
    else:
        residual = tl.load(X+i,i<NUMEL,0)
        scale = tl.load(S+i%N,i<NUMEL,0)
        acc = _add(residual,_mul(acc,scale))
    tl.store(Y+i,acc,i<NUMEL)


def linear(x, packed, mode, residual, scale, library, *, stages=2):
    """Internal matrix dispatch; library must come from math_library()."""
    expected = (1280,5120) if mode == 'gelu' else (5120,1280) if mode == 'residual' else None
    if (expected is None or packed.ndim != 2 or tuple(packed.shape) != expected
            or tuple(x.shape) != (24,expected[0]) or not x.is_cuda or x.dtype != torch.float32
            or packed.device != x.device or packed.dtype != x.dtype
            or not x.is_contiguous() or not packed.is_contiguous()):
        raise ValueError('Expected a supported contiguous CUDA FP32 FFN matrix')
    k,n = expected
    if mode == 'residual' and (residual is None or scale is None or residual.shape != (24,n)
            or scale.shape != (n,) or any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous()
                                        for t in (residual,scale))):
        raise ValueError('Expected matching contiguous FP32 residual and scale')
    partials = torch.empty((k//256,24,n),device=x.device,dtype=x.dtype)
    out = torch.empty((24,n),device=x.device,dtype=x.dtype)
    _partials[(1,n//128,k//256)](x,packed,partials,24,n,k,256,32,128,32,num_warps=4,
                                    num_stages=stages,enable_fp_fusion=False)
    _ffn_reduce[(24*n//256,)](partials,residual if residual is not None else partials,
        scale if scale is not None else partials,out,24*n,n,k//256,mode,256,
        enable_fp_fusion=False,extern_libs={'libdevice':library})
    return out


def observed(layer):
    return (module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
            or any(m._forward_hooks or m._forward_pre_hooks for m in
                   (layer.norm2,layer.linear1,layer.linear2,layer.layer_scale_2)))


def forward(self, x):
    runtime = self._fast_ffn_runtime
    if observed(self):
        return self._fast_observed_ffn(x)
    if (not runtime.ffn_enabled or runtime.backend != 'triton' or self.activation is not F.gelu or self.gating is not None
            or self.weights_per_step or type(self.norm2) is not torch.nn.LayerNorm
            or 'forward' in self.norm2.__dict__
            or x.ndim < 2 or x.shape[-1] != 1280 or x.numel() != 24*1280
            or x.dtype != torch.float32 or x.device != runtime.device or x.requires_grad
            or not x.is_contiguous() or torch.is_autocast_enabled('cuda')):
        return self._fast_original_ffn(x)
    normalized = self.norm2(x)
    # A norm forward can install hooks or replace the activation. Recheck before
    # fusing, and reuse its output so an observer is not called twice.
    if observed(self) or self.activation is not F.gelu:
        update = self.linear2(self.activation(self.linear1(normalized)))
        return x.to(update) + self.layer_scale_2(update)
    if not self._fast_ffn_fuse:
        hidden = self.activation(self.linear1(normalized,_fast_stages=2))
        update = self.linear2(hidden,_fast_stages=2)
        return self._fast_scale_add(x,update,self.layer_scale_2.scale)
    hidden = self.linear1(normalized,_fast_epilogue=('gelu',None,None,self._fast_ffn_library))
    return self.linear2(hidden,_fast_epilogue=('residual',x,self.layer_scale_2.scale,self._fast_ffn_library))

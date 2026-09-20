"""Research-only native FP32 GEMV with exact GELU or residual epilogue."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from fast_moss.ffn import _mul,_add,math_library
from fast_moss.small_matrices import CONFIGS


@triton.jit
def _fused(X,W,R,S,Y,N:tl.constexpr,K:tl.constexpr,L:tl.constexpr,
           BN:tl.constexpr,U:tl.constexpr,MODE:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,L)
    acc=tl.full((BN,L),0,tl.float32)
    for block in tl.range(tl.cdiv(K,L),loop_unroll_factor=U):
        k=block*L+lane
        x=tl.load(X+k,k<K,0)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
        acc=tl.fma(x[None,:],w,acc)
    for i in tl.static_range(0,tl.constexpr(L.bit_length()-1)):
        delta=L//2 >> i
        index=tl.broadcast_to(((lane+delta)%L)[None,:],(BN,L))
        acc=acc+tl.gather(acc,index,1)
    out=tl.reshape(tl.gather(acc,tl.full((BN,1),0,tl.int32),1),(BN,))
    out=_add(out,0.0)
    if MODE=='gelu':
        out=_mul(_mul(out,0.5),_add(libdevice.erf(_mul(out,0.7071067811865476)),1.0))
    else:
        residual=tl.load(R+n,n<N,0)
        scale=tl.load(S+n,n<N,0)
        out=_add(residual,_mul(out,scale))
    tl.store(Y+n,out,n<N)


def linear(x,w,mode,residual=None,scale=None,library=None,config=None,resources=False):
    n,k=w.shape;shape=(1,n,k)
    if (shape not in CONFIGS or mode not in ('gelu','residual') or x.shape!=(1,k)
            or not x.is_cuda or x.dtype!=torch.float32 or w.dtype!=x.dtype or w.device!=x.device
            or not x.is_contiguous() or not w.is_contiguous()):
        raise ValueError('Expected supported native CUDA FP32 GEMV')
    if mode=='residual' and (residual is None or scale is None or residual.shape!=(1,n) or scale.shape!=(n,)
            or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in (residual,scale))):
        raise ValueError('Expected matching residual and scale')
    _,lanes,bn,warps,u=CONFIGS[shape] if config is None else config
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    kernel=_fused[(triton.cdiv(n,bn),)](x,w,residual if residual is not None else x,
        scale if scale is not None else x,out,n,k,lanes,bn,u,mode,num_warps=warps,
        enable_fp_fusion=False,extern_libs={'libdevice':library or math_library()})
    if resources:
        ptx=kernel.asm['ptx']
        return out,{'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
            'matrix_instructions':sum(ptx.count(s) for s in ('mma.sync','wgmma.','tcgen05.mma'))}
    return out

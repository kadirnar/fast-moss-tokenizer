"""Research-only exact FP32 GEMV with output-group interleaved weight storage."""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@dataclass
class Packed:
    data: torch.Tensor
    n: int
    k: int
    lanes: int
    group: int


def pack(w,lanes,group):
    if (w.ndim!=2 or w.dtype!=torch.float32 or not w.is_cuda
            or lanes not in (8,16,32) or group not in (1,2,4,8,16,32)):
        raise ValueError('Expected CUDA FP32 weights and supported lane/group sizes')
    n,k=w.shape
    if not n or not k or n%group or k%lanes:
        raise ValueError('Expected complete output groups and cyclic lanes')
    # Integer views explicitly preserve every IEEE payload bit. This is a copy,
    # not a mutation of model parameters or a reduced-precision representation.
    data=w.contiguous().view(torch.int32).reshape(n//group,group,k//lanes,lanes)
    data=data.permute(0,2,1,3).contiguous().clone() if group==1 else data.permute(0,2,1,3).contiguous()
    return Packed(data.view(torch.float32),n,k,lanes,group)


def unpack(p):
    return p.data.view(torch.int32).permute(0,2,1,3).contiguous().reshape(p.n,p.k).view(torch.float32)


@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,L:tl.constexpr,G:tl.constexpr,U:tl.constexpr):
    block=tl.program_id(0)
    row=tl.arange(0,G)
    lane=tl.arange(0,L)
    acc=tl.full((G,L),0,tl.float32)
    for step in tl.range(K//L,loop_unroll_factor=U):
        x=tl.load(X+step*L+lane)
        w=tl.load(W+block*G*K+step*G*L+row[:,None]*L+lane[None,:])
        acc=tl.fma(x[None,:],w,acc)
    for i in tl.static_range(0,tl.constexpr(L.bit_length()-1)):
        delta=L//2 >> i
        index=tl.broadcast_to(((lane+delta)%L)[None,:],(G,L))
        acc=acc+tl.gather(acc,index,1)
    out=tl.reshape(tl.gather(acc,tl.full((G,1),0,tl.int32),1),(G,))
    out=tl.inline_asm_elementwise('add.rn.f32 $0, $1, 0f00000000;',
        constraints='=f,f',args=[out],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+block*G+row,out)


def gemv(x,p,warps=1,unroll=4,resources=False):
    if (x.shape!=(1,p.k) or x.dtype!=torch.float32 or x.device!=p.data.device
            or not x.is_contiguous() or warps not in (1,2,4) or unroll not in (4,16,32)):
        raise ValueError('Expected contiguous CUDA FP32 GEMV and supported launch parameters')
    out=torch.empty((1,p.n),device=x.device,dtype=x.dtype)
    kernel=_gemv[(p.n//p.group,)](x,p.data,out,p.n,p.k,p.lanes,p.group,unroll,
                              num_warps=warps,enable_fp_fusion=False)
    if resources:
        ptx=kernel.asm['ptx']
        return out,{'registers':kernel.n_regs,'spills':kernel.n_spills,
            'shared_bytes':kernel.metadata.shared,'ptx_fma':ptx.count('fma.rn.f32'),
            'matrix_instructions':sum(ptx.count(s) for s in ('mma.sync','wgmma.','tcgen05.mma'))}
    return out

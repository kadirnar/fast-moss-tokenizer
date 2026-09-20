"""Experimental FP32 split-cache attention. Not enabled in the supported runtime.

Reduction order differs from PyTorch SDPA. No reduced-precision operands or TF32.
"""
import torch
import triton
import triton.language as tl


# Keep bias row strides dynamic so padding/densification cannot change the
# compiler-selected reduction tree (notably for 256-key blocks).
@triton.jit(do_not_specialize=["A0", "A2"])
def _partial(Q,K,V,BIAS,PART,STATS,H:tl.constexpr,T:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
             Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
             K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
             V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,V3:tl.constexpr,
             A0,A2,A3:tl.constexpr,
             SPLITS:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0);split=tl.program_id(1)
    t=row%T;h=row//T%H;b=row//(T*H)
    s=split*BK+tl.arange(0,BK);d=tl.arange(0,BD)
    q=tl.load(Q+b*Q0+h*Q1+t*Q2+d*Q3,d<D,0)
    k=tl.load(K+b*K0+h*K1+s[:,None]*K2+d[None,:]*K3,(s[:,None]<S)&(d[None,:]<D),0)
    score=tl.sum(k*q[None,:],axis=1)*(D**-0.5)
    bias=tl.load(BIAS+b*A0+t*A2+s*A3,s<S,-float('inf'))
    score=score+bias
    maximum=tl.max(score,axis=0)
    safe_max=tl.where(maximum==-float('inf'),0.,maximum)
    p=tl.exp(score-safe_max)
    denom=tl.sum(p,axis=0)
    v=tl.load(V+b*V0+h*V1+s[:,None]*V2+d[None,:]*V3,(s[:,None]<S)&(d[None,:]<D),0)
    acc=tl.sum(p[:,None]*v,axis=0)
    tl.store(PART+(row*SPLITS+split)*D+d,acc,d<D)
    tl.store(STATS+(row*SPLITS+split)*2,maximum)
    tl.store(STATS+(row*SPLITS+split)*2+1,denom)


@triton.jit
def _merge(PART,STATS,OUT,D:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)
    s=tl.arange(0,BS);d=tl.arange(0,BD)
    maximum=tl.load(STATS+(row*SPLITS+s)*2,s<SPLITS,-float('inf'))
    denom=tl.load(STATS+(row*SPLITS+s)*2+1,s<SPLITS,0)
    total_max=tl.max(maximum,axis=0)
    total_max=tl.where(total_max==-float('inf'),0.,total_max)
    scale=tl.exp(maximum-total_max)
    norm=tl.sum(denom*scale,axis=0)
    acc=tl.load(PART+(row*SPLITS+s[:,None])*D+d[None,:],(s[:,None]<SPLITS)&(d[None,:]<D),0)
    value=tl.sum(acc*scale[:,None],axis=0)/tl.where(norm>0,norm,1.)
    tl.store(OUT+row*D+d,value,d<D)


def split_attention(q,k,v,bias,block=64):
    b,h,t,d=q.shape;s=k.shape[2]
    if (s<1 or t<1 or block<1 or block&(block-1)
            or q.dtype!=torch.float32 or k.dtype!=torch.float32 or v.dtype!=torch.float32
            or bias.dtype!=torch.float32 or bias.shape!=(b,1,t,s)
            or k.shape!=v.shape or k.shape[:2]!=(b,h) or k.shape[-1]!=d
            or not all(x.is_cuda and x.device==q.device for x in [q,k,v,bias])):
        raise ValueError('Expected CUDA FP32 attention inputs and additive (B,1,T,S) bias')
    # Fix reduction layout across eager calls and graph input clones. Triton
    # can otherwise choose a different reduction tree for gapped feature strides.
    # Actual model Q/K/V buffers are already contiguous; those incur no copy.
    q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
    splits=triton.cdiv(s,block)
    partial=torch.empty((b*h*t,splits,d),device=q.device,dtype=q.dtype)
    stats=torch.empty((b*h*t,splits,2),device=q.device,dtype=q.dtype)
    out=torch.empty((b,h,t,d),device=q.device,dtype=q.dtype)
    _partial[(b*h*t,splits)](q,k,v,bias,partial,stats,h,t,s,d,*q.stride(),*k.stride(),*v.stride(),
                            bias.stride(0),bias.stride(2),bias.stride(3),splits,block,triton.next_power_of_2(d),
                            num_warps=4,enable_fp_fusion=False)
    _merge[(b*h*t,)](partial,stats,out,d,splits,triton.next_power_of_2(splits),triton.next_power_of_2(d),
                    num_warps=4,enable_fp_fusion=False)
    return out

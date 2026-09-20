"""Pinned native-layout small-row FP32 matrices with ordered tiled reductions.

Each lane resets its FMA accumulator every 256 K terms. Tile sums precede
serial lane reduction; interchanging those reductions changes rounding.
Validated only for the recorded GPU/compiler/model through MatrixRuntime.
"""
import torch
import triton
import triton.language as tl


# (M,N,K): (strategy, rows per block, columns per block, warps, unroll).
CONFIGS = {
    (3, 5120, 1280): ('grouped', 4, 8, 4, 4),
    (3, 1280, 5120): ('split', 4, 8, 4, 16),
    (12, 768, 3072): ('split', 4, 8, 4, 16),
}
SHAPES = set(CONFIGS)


@triton.jit
def _small_grouped(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,UNROLL:tl.constexpr):
    m=tl.program_id(1)*BM+tl.arange(0,BM)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,16)
    total=tl.full((BM,BN,16),0,tl.float32)
    for start in range(tl.cdiv(K,256)):
        acc=tl.full((BM,BN,16),0,tl.float32)
        for step in tl.range(16,loop_unroll_factor=UNROLL):
            k=start*256+step*16+lane
            x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
            w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
            acc=tl.fma(x[:,None,:],w[None,:,:],acc)
        total=total+acc
    out=tl.reshape(tl.gather(total,tl.full((BM,BN,1),0,tl.int32),2),(BM,BN))
    for i in tl.static_range(1,16):
        out=out+tl.reshape(tl.gather(total,tl.full((BM,BN,1),i,tl.int32),2),(BM,BN))
    tl.store(Y+m[:,None]*N+n[None,:],out,(m[:,None]<M)&(n[None,:]<N))

@triton.jit
def _small_parts(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr):
    m=tl.program_id(1)*BM+tl.arange(0,BM)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,16)
    part=tl.program_id(2)
    acc=tl.full((BM,BN,16),0,tl.float32)
    for step in tl.static_range(16):
        k=part*256+step*16+lane
        x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
        acc=tl.fma(x[:,None,:],w[None,:,:],acc)
    offset=((part*M+m[:,None,None])*N+n[None,:,None])*16+lane[None,None,:]
    tl.store(P+offset,acc,(m[:,None,None]<M)&(n[None,:,None]<N))

@triton.jit
def _small_reduce(P,Y,MN:tl.constexpr,PARTS:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    lane=tl.arange(0,16)
    total=tl.full((B,16),0,tl.float32)
    for part in tl.static_range(PARTS):
        total=total+tl.load(P+(part*MN+r[:,None])*16+lane[None,:],r[:,None]<MN,0)
    out=tl.reshape(tl.gather(total,tl.full((B,1),0,tl.int32),1),(B,))
    for i in tl.static_range(1,16):
        out=out+tl.reshape(tl.gather(total,tl.full((B,1),i,tl.int32),1),(B,))
    tl.store(Y+r,out,r<MN)


def linear(x, weight):
    """Supported contiguous X=(M,K), native-layout W=(N,K)."""
    message = 'Expected a supported contiguous CUDA FP32 small matrix shape'
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(message)
    m, k = x.shape
    n = weight.shape[0]
    shape = (m,n,k)
    if (weight.shape[1] != k or shape not in CONFIGS or not x.is_cuda
            or x.dtype != torch.float32 or weight.dtype != x.dtype or weight.device != x.device
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError(message)
    strategy, bm, bn, warps, unroll = CONFIGS[shape]
    out = torch.empty((m,n),device=x.device,dtype=x.dtype)
    if strategy == 'grouped':
        _small_grouped[(triton.cdiv(n,bn),triton.cdiv(m,bm))](
            x,weight,out,m,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)
    else:
        parts = triton.cdiv(k,256)
        partials = torch.empty((parts,m,n,16),device=x.device,dtype=x.dtype)
        _small_parts[(triton.cdiv(n,bn),triton.cdiv(m,bm),parts)](
            x,weight,partials,m,n,k,bm,bn,num_warps=warps,enable_fp_fusion=False)
        _small_reduce[(triton.cdiv(m*n,32),)](
            partials,out,m*n,parts,32,num_warps=4,enable_fp_fusion=False)
    return out

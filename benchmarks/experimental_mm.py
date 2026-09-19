"""Exploratory FP32 GEMM; deliberately not connected to the inference runtime."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+kk
        a=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0)
        b=tl.load(W+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0)
        acc=tl.dot(a,b,acc,input_precision="ieee")
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def linear(x,weight,tile=(16,32,32)):
    if (x.ndim!=2 or weight.ndim!=2 or x.shape[1]!=weight.shape[1]
            or not x.is_contiguous() or not weight.is_contiguous()
            or x.dtype!=torch.float32 or weight.dtype!=torch.float32
            or not x.is_cuda or x.device!=weight.device):
        raise ValueError("Expected contiguous compatible CUDA FP32 matrices")
    m,k=x.shape;n=weight.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    bm,bn,bk=tile
    _mm[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,weight,out,m,n,k,bm,bn,bk,num_warps=4)
    return out


@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(1)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    x=tl.load(X+row*K+k,k<K,0)
    w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
    y=tl.sum(w*x[None,:],axis=1)
    tl.store(Y+row*N+n,y,n<N)


def gemv_rows(x,weight,rows_per_block=2):
    m,k=x.shape;n=weight.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _gemv[(triton.cdiv(n,rows_per_block),m)](x,weight,out,n,k,rows_per_block,triton.next_power_of_2(k),
                                          num_warps=8 if k>2048 else 4,enable_fp_fusion=False)
    return out

"""Research-only component-product tensor-core GEMM with FP32 storage/output.

TF32x3 recovers much of the input precision through correction products. It is
not bitwise FP32 SIMT. BF16x6 uses three components and six products with a
separate low-product accumulator. Neither mode establishes no quality loss.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,MODE:tl.constexpr):
    # Group rows to reuse weight tiles in L2.
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM);nn=tl.cdiv(N,BN);group=8
    group_id=pid//(group*nn);first=group_id*group
    size=tl.minimum(nm-first,group)
    pm=first+pid%size;pn=(pid%(group*nn))//size
    rows=pm*BM+tl.arange(0,BM)
    cols=pn*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    correction=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+kk
        a=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0)
        b=tl.load(W+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0)
        if MODE=='tf32x3':
            acc=tl.dot(a,b,acc,input_precision='tf32x3')
        else:
            ah=a.to(tl.bfloat16);bh=b.to(tl.bfloat16)
            ar=a-ah.to(tl.float32);br=b-bh.to(tl.float32)
            am=ar.to(tl.bfloat16);bm=br.to(tl.bfloat16)
            al=(ar-am.to(tl.float32)).to(tl.bfloat16)
            bl=(br-bm.to(tl.float32)).to(tl.bfloat16)
            correction=tl.dot(al,bh,correction)
            correction=tl.dot(am,bm,correction)
            correction=tl.dot(ah,bl,correction)
            correction=tl.dot(am,bh,correction)
            correction=tl.dot(ah,bm,correction)
            block=tl.dot(ah,bh)
            # Force the high-product block sum through CUDA round-to-nearest
            # addition, not the tensor-core accumulator's internal rounding.
            acc=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',
                                         args=[acc,block],dtype=tl.float32,is_pure=True,pack=1)
    if MODE=='bf16x6':acc=acc+correction
    tl.store(Y+rows[:,None]*N+cols[None,:],acc,(rows[:,None]<M)&(cols[None,:]<N))


def linear(x,weight,tile=(32,64,32,4,3),return_kernel=False,mode='tf32x3'):
    if (x.ndim!=2 or weight.ndim!=2 or x.shape[1]!=weight.shape[1] or not x.is_contiguous()
            or not weight.is_contiguous() or x.dtype!=torch.float32 or weight.dtype!=torch.float32
            or not x.is_cuda or weight.device!=x.device):
        raise ValueError('Expected compatible contiguous CUDA FP32 input and weight matrices')
    if mode not in ['tf32x3','bf16x6']:raise ValueError('Unknown component-product mode')
    m,k=x.shape;n=weight.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    bm,bn,bk,warps,stages=tile
    kernel=_mm[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](x,weight,out,m,n,k,bm,bn,bk,mode,
                 num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out

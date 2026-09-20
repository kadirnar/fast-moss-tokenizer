"""Triton single-program native K-tile combination, no global partial buffer."""
import torch
import triton
import triton.language as tl


@triton.jit
def _cta(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         BM:tl.constexpr,BN:tl.constexpr,UNROLL:tl.constexpr):
    m=tl.program_id(1)*BM+tl.arange(0,BM)
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    index=tl.arange(0,64);part=index//16;lane=index%16
    acc=tl.full((BM,BN,64),0,tl.float32)
    for step in tl.range(16,loop_unroll_factor=UNROLL):
        k=part*256+step*16+lane
        x=tl.load(X+m[:,None]*K+k[None,:],(m[:,None]<M)&(part[None,:]<3),0)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(part[None,:]<3),0)
        acc=tl.fma(x[:,None,:],w[None,:,:],acc)
    lanes=tl.arange(0,16)
    total=tl.full((BM,BN,16),0,tl.float32)
    for p in tl.static_range(3):
        value=tl.gather(acc,tl.broadcast_to((lanes+16*p)[None,None,:],(BM,BN,16)),2)
        total=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',
            args=[total,value],dtype=tl.float32,is_pure=True,pack=1)
    out=tl.reshape(tl.gather(total,tl.full((BM,BN,1),0,tl.int32),2),(BM,BN))
    for i in tl.static_range(1,16):
        value=tl.reshape(tl.gather(total,tl.full((BM,BN,1),i,tl.int32),2),(BM,BN))
        out=out+value
    tl.store(Y+m[:,None]*N+n[None,:],out,(m[:,None]<M)&(n[None,:]<N))


def linear(x,w,config,resources=False):
    bm,bn,warps,u=config
    if (x.ndim!=2 or w.ndim!=2 or x.shape[1]!=768 or w.shape[1]!=768
            or not x.is_cuda or x.dtype!=torch.float32 or w.dtype!=x.dtype or w.device!=x.device
            or not x.is_contiguous() or not w.is_contiguous() or min(x.shape[0],w.shape[0])<1
            or bm not in (4,8) or bn not in (2,4,8) or warps not in (2,4,8) or u not in (4,16)):
        raise ValueError('Expected contiguous CUDA FP32 K=768 operands and validated launch dimensions')
    m,k=x.shape;n=w.shape[0]
    with torch.cuda.device(x.device):
        out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        kernel=_cta[(triton.cdiv(n,bn),triton.cdiv(m,bm))](x,w,out,m,n,k,bm,bn,u,num_warps=warps,enable_fp_fusion=False)
    return (out,{'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}) if resources else out

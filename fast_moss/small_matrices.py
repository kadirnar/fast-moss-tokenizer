"""Pinned native-layout small-row FP32 matrices with ordered tiled reductions.

Multi-row lanes reset their FMA accumulator every 256 K terms, then sum
tiles before serial lane reduction. GEMV uses full-K cyclic lanes and an
explicit halving reduction, followed by a rounding-preserving add of +0.
Validated only for the recorded GPU/compiler/model through MatrixRuntime.
"""
import torch
import triton
import triton.language as tl


# (M,N,K): (strategy, rows per block (GEMV: lanes), columns per block, warps, unroll).
CONFIGS = {
    (1, 768, 1280): ('gemv', 32, 4, 2, 32),
    (1, 1280, 768): ('gemv', 16, 4, 2, 32),
    (1, 1280, 1280): ('gemv', 16, 4, 2, 4),
    (1, 1280, 5120): ('gemv', 16, 4, 1, 32),
    (1, 3840, 1280): ('gemv', 8, 8, 1, 32),
    (1, 5120, 1280): ('gemv', 8, 8, 1, 32),
    (2, 640, 768): ('fixed', 2, 4, 1, 4),
    (2, 768, 640): ('fixed', 2, 4, 1, 4),
    (2, 768, 768): ('fixed', 2, 4, 1, 4),
    (2, 768, 3072): ('split', 2, 4, 2, 16),
    (2, 2304, 768): ('fixed', 2, 4, 1, 16),
    (2, 3072, 768): ('fixed', 2, 4, 1, 4),
    (3, 768, 1280): ('fixed', 3, 4, 1, 16),
    (3, 1280, 768): ('fixed', 3, 4, 1, 16),
    (3, 1280, 1280): ('fixed', 3, 4, 1, 16),
    (3, 1280, 5120): ('split', 4, 8, 4, 16),
    (3, 3840, 1280): ('fixed', 3, 4, 1, 16),
    (3, 5120, 1280): ('fixed', 3, 4, 1, 16),
    (4, 768, 3072): ('split', 4, 8, 4, 16),
    (6, 640, 768): ('fixed', 3, 4, 1, 16),
    (6, 768, 768): ('fixed', 3, 4, 1, 16),
    (6, 768, 3072): ('fixed', 8, 4, 2, 16),
    (8, 240, 768): ('fixed', 3, 4, 1, 16),
    (8, 384, 768): ('fixed', 3, 4, 1, 16),
    (8, 768, 240): ('fixed', 4, 4, 1, 16),
    (8, 768, 384): ('fixed', 3, 4, 2, 4),
    (8, 768, 768): ('fixed', 4, 4, 1, 4),
    (8, 768, 1280): ('split', 4, 8, 4, 16),
    (8, 768, 3072): ('split', 4, 8, 4, 16),
    (8, 1280, 768): ('fixed', 4, 4, 1, 4),
    (8, 1280, 1280): ('fixed', 3, 4, 1, 16),
    (8, 1280, 5120): ('fixed', 3, 4, 1, 16),
    (8, 3072, 768): ('fixed', 4, 4, 1, 4),
    (8, 3840, 1280): ('fixed', 4, 4, 1, 4),
    (8, 5120, 1280): ('fixed', 4, 4, 1, 4),
    (12, 384, 768): ('fixed', 3, 4, 1, 16),
    (12, 768, 3072): ('fixed', 3, 4, 1, 16),
    (12, 3072, 768): ('fixed', 3, 4, 1, 16),
}

SHAPES = set(CONFIGS)


@triton.jit
def _small_gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,LANES:tl.constexpr,
          BN:tl.constexpr,UNROLL:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,LANES)
    acc=tl.full((BN,LANES),0,tl.float32)
    for block in tl.range(tl.cdiv(K,LANES),loop_unroll_factor=UNROLL):
        k=block*LANES+lane
        x=tl.load(X+k,k<K,0)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
        acc=tl.fma(x[None,:],w,acc)
    for i in tl.static_range(0,tl.constexpr(LANES.bit_length()-1)):
        delta=LANES//2 >> i
        index=tl.broadcast_to(((lane+delta)%LANES)[None,:],(BN,LANES))
        acc=acc+tl.gather(acc,index,1)
    out=tl.reshape(tl.gather(acc,tl.full((BN,1),0,tl.int32),1),(BN,))
    # Preserve native positive zero after negative subnormal underflow.
    # Plain `out + 0` can be optimized away by the compiler.
    out=tl.inline_asm_elementwise('add.rn.f32 $0, $1, 0f00000000;',
        constraints='=f,f',args=[out],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+n,out,n<N)


# Separate row accumulators avoid imposing a power-of-two tensor row axis.
# Selected row tiles balance weight reuse, occupancy and masked rows.
@triton.jit
def _small_fixed(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           ROWS:tl.constexpr,BN:tl.constexpr,UNROLL:tl.constexpr):
    base=tl.program_id(1)*ROWS
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,16)
    if ROWS>0:
        total0=tl.full((BN,16),0,tl.float32)
    if ROWS>1:
        total1=tl.full((BN,16),0,tl.float32)
    if ROWS>2:
        total2=tl.full((BN,16),0,tl.float32)
    if ROWS>3:
        total3=tl.full((BN,16),0,tl.float32)
    if ROWS>4:
        total4=tl.full((BN,16),0,tl.float32)
    if ROWS>5:
        total5=tl.full((BN,16),0,tl.float32)
    if ROWS>6:
        total6=tl.full((BN,16),0,tl.float32)
    if ROWS>7:
        total7=tl.full((BN,16),0,tl.float32)
    for start in range(tl.cdiv(K,256)):
        if ROWS>0:
            acc0=tl.full((BN,16),0,tl.float32)
        if ROWS>1:
            acc1=tl.full((BN,16),0,tl.float32)
        if ROWS>2:
            acc2=tl.full((BN,16),0,tl.float32)
        if ROWS>3:
            acc3=tl.full((BN,16),0,tl.float32)
        if ROWS>4:
            acc4=tl.full((BN,16),0,tl.float32)
        if ROWS>5:
            acc5=tl.full((BN,16),0,tl.float32)
        if ROWS>6:
            acc6=tl.full((BN,16),0,tl.float32)
        if ROWS>7:
            acc7=tl.full((BN,16),0,tl.float32)
        for step in tl.range(16,loop_unroll_factor=UNROLL):
            k=start*256+step*16+lane
            w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
            if ROWS>0:
                x0=tl.load(X+(base+0)*K+k,(base+0<M)&(k<K),0)
                acc0=tl.fma(x0[None,:],w,acc0)
            if ROWS>1:
                x1=tl.load(X+(base+1)*K+k,(base+1<M)&(k<K),0)
                acc1=tl.fma(x1[None,:],w,acc1)
            if ROWS>2:
                x2=tl.load(X+(base+2)*K+k,(base+2<M)&(k<K),0)
                acc2=tl.fma(x2[None,:],w,acc2)
            if ROWS>3:
                x3=tl.load(X+(base+3)*K+k,(base+3<M)&(k<K),0)
                acc3=tl.fma(x3[None,:],w,acc3)
            if ROWS>4:
                x4=tl.load(X+(base+4)*K+k,(base+4<M)&(k<K),0)
                acc4=tl.fma(x4[None,:],w,acc4)
            if ROWS>5:
                x5=tl.load(X+(base+5)*K+k,(base+5<M)&(k<K),0)
                acc5=tl.fma(x5[None,:],w,acc5)
            if ROWS>6:
                x6=tl.load(X+(base+6)*K+k,(base+6<M)&(k<K),0)
                acc6=tl.fma(x6[None,:],w,acc6)
            if ROWS>7:
                x7=tl.load(X+(base+7)*K+k,(base+7<M)&(k<K),0)
                acc7=tl.fma(x7[None,:],w,acc7)
        if ROWS>0:
            total0=total0+acc0
        if ROWS>1:
            total1=total1+acc1
        if ROWS>2:
            total2=total2+acc2
        if ROWS>3:
            total3=total3+acc3
        if ROWS>4:
            total4=total4+acc4
        if ROWS>5:
            total5=total5+acc5
        if ROWS>6:
            total6=total6+acc6
        if ROWS>7:
            total7=total7+acc7
    if ROWS>0:
        out0=tl.reshape(tl.gather(total0,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out0=out0+tl.reshape(tl.gather(total0,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+0)*N+n,out0,(base+0<M)&(n<N))
    if ROWS>1:
        out1=tl.reshape(tl.gather(total1,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out1=out1+tl.reshape(tl.gather(total1,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+1)*N+n,out1,(base+1<M)&(n<N))
    if ROWS>2:
        out2=tl.reshape(tl.gather(total2,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out2=out2+tl.reshape(tl.gather(total2,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+2)*N+n,out2,(base+2<M)&(n<N))
    if ROWS>3:
        out3=tl.reshape(tl.gather(total3,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out3=out3+tl.reshape(tl.gather(total3,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+3)*N+n,out3,(base+3<M)&(n<N))
    if ROWS>4:
        out4=tl.reshape(tl.gather(total4,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out4=out4+tl.reshape(tl.gather(total4,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+4)*N+n,out4,(base+4<M)&(n<N))
    if ROWS>5:
        out5=tl.reshape(tl.gather(total5,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out5=out5+tl.reshape(tl.gather(total5,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+5)*N+n,out5,(base+5<M)&(n<N))
    if ROWS>6:
        out6=tl.reshape(tl.gather(total6,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out6=out6+tl.reshape(tl.gather(total6,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+6)*N+n,out6,(base+6<M)&(n<N))
    if ROWS>7:
        out7=tl.reshape(tl.gather(total7,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out7=out7+tl.reshape(tl.gather(total7,tl.full((BN,1),j,tl.int32),1),(BN,))
        tl.store(Y+(base+7)*N+n,out7,(base+7<M)&(n<N))


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
    if strategy == 'gemv':
        _small_gemv[(triton.cdiv(n,bn),)](
            x,weight,out,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)
    elif strategy == 'fixed':
        _small_fixed[(triton.cdiv(n,bn),triton.cdiv(m,bm))](
            x,weight,out,m,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)
    elif strategy == 'grouped':
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

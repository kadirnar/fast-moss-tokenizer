"""Research-only exact small GEMM with explicit, unpadded row accumulators."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph
from benchmarks.compare import difference
from benchmarks.ffn_resources import resources
from fast_moss.loading import strict_precision,REVISION

@triton.jit
def _fixed(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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


def fixed(x,w,rows,bn,warps,unroll,return_kernel=False):
    m,k=x.shape;n=w.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    kernel=_fixed[(triton.cdiv(n,bn),triton.cdiv(m,rows))](x,w,out,m,n,k,rows,bn,unroll,
                                                 num_warps=warps,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only unpadded row accumulator search, warm components',
            'revision':REVISION,'torch':torch.__version__,'triton':triton.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape in [(6,3072,768),(6,768,3072),(3,3840,1280),(12,2304,768)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        record={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=20),'trials':[]}
        for rows in ([3,4] if shape[0]==3 else [3,6,8]):
            for bn in [4,8,16]:
                for warps in [1,2,4]:
                    for unroll in [4,16]:
                        config=(rows,bn,warps,unroll)
                        fn=lambda:fixed(x,w,*config)
                        out,kernel=fixed(x,w,*config,return_kernel=True)
                        d=difference(ref,out)
                        ms=do_bench_cudagraph(fn,rep=10)
                        record['trials'].append({'config':config,'difference':d,'ms':ms,'resources':resources(kernel)})
        report['records'].append(record)
        print(shape,'native',record['native_ms'],'best',min(record['trials'],key=lambda t:t['ms']),flush=True)
        Path('results/small_fixed_rows.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

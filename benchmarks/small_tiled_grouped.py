"""Research-only row-sharing version of exact tiled cyclic small GEMM."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph
from benchmarks.compare import difference
from fast_moss.loading import strict_precision

@triton.jit
def _grouped(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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


def grouped(x,w,bm=4,bn=8,warps=4,unroll=16):
    m,k=x.shape;n=w.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _grouped[(triton.cdiv(n,bn),triton.cdiv(m,bm))](x,w,out,m,n,k,bm,bn,unroll,
                                                            num_warps=warps,enable_fp_fusion=False)
    return out

@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only row sharing component search; warm timings','records':[]}
    for shape in [(3,5120,1280),(3,1280,5120),(6,3072,768),(12,768,3072)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        r={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=40),'trials':[]}
        for bm in (1,4,8):
            for bn in (4,8,16):
                for warps in (1,2,4):
                    for unroll in (4,16):
                        fn=lambda:grouped(x,w,bm,bn,warps,unroll)
                        d=difference(ref,fn());ms=do_bench_cudagraph(fn,rep=20)
                        r['trials'].append({'config':[bm,bn,warps,unroll],'difference':d,'ms':ms})
        report['records'].append(r)
        print(shape,'native',r['native_ms'],'best',min(r['trials'],key=lambda t:t['ms']),flush=True)
        Path('results/small_tiled_grouped.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

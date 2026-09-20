"""Research-only split-tile small GEMM with ordered lane/tile reductions."""
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
def _parts(X,W,P,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
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
def _reduce(P,Y,MN:tl.constexpr,PARTS:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0)*B+tl.arange(0,B)
    lane=tl.arange(0,16)
    total=tl.full((B,16),0,tl.float32)
    for part in tl.static_range(PARTS):
        total=total+tl.load(P+(part*MN+r[:,None])*16+lane[None,:],r[:,None]<MN,0)
    out=tl.reshape(tl.gather(total,tl.full((B,1),0,tl.int32),1),(B,))
    for i in tl.static_range(1,16):
        out=out+tl.reshape(tl.gather(total,tl.full((B,1),i,tl.int32),1),(B,))
    tl.store(Y+r,out,r<MN)


def split(x,w,bm=4,bn=8,warps=4):
    m,k=x.shape;n=w.shape[0];parts=triton.cdiv(k,256)
    p=torch.empty((parts,m,n,16),device=x.device,dtype=x.dtype)
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _parts[(triton.cdiv(n,bn),triton.cdiv(m,bm),parts)](x,w,p,m,n,k,bm,bn,
                                               num_warps=warps,enable_fp_fusion=False)
    _reduce[(triton.cdiv(m*n,32),)](p,out,m*n,parts,32,num_warps=4,enable_fp_fusion=False)
    return out

@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only split tile component search; warm timings','records':[]}
    for shape in [(3,5120,1280),(3,1280,5120),(6,3072,768),(12,768,3072)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        r={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=30),'trials':[]}
        for bm in (1,4,8):
            for bn in (4,8,16):
                for warps in (1,2,4):
                    fn=lambda:split(x,w,bm,bn,warps)
                    d=difference(ref,fn());ms=do_bench_cudagraph(fn,rep=15)
                    r['trials'].append({'config':[bm,bn,warps],'difference':d,'ms':ms})
        report['records'].append(r)
        print(shape,'native',r['native_ms'],'best',min(r['trials'],key=lambda t:t['ms']),flush=True)
        Path('results/small_tiled_split.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

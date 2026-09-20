"""Research-only tiled cyclic FP32 accumulation for small-row vendor matrices."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from benchmarks.small_arithmetic import _fold
from benchmarks.compare import difference
from fast_moss.loading import strict_precision


@triton.jit
def _tiled(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           CHUNK:tl.constexpr,LANES:tl.constexpr,BN:tl.constexpr,ORDER:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    m=tl.program_id(1)
    lane=tl.arange(0,LANES)
    total=tl.full((BN,LANES),0,tl.float32)
    for start in range(tl.cdiv(K,CHUNK)):
        acc=tl.full((BN,LANES),0,tl.float32)
        for step in range(CHUNK//LANES):
            k=start*CHUNK+step*LANES+lane
            x=tl.load(X+m*K+k,k<K,0)
            w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
            acc=tl.fma(x[None,:],w,acc)
        total=total+acc
    out=tl.reshape(_fold(total[:,:,None],1,LANES,ORDER),(BN,))
    tl.store(Y+m*N+n,out,n<N)


def tiled(x,w,chunk=256,lanes=16,bn=8,order='serial',n=None,warps=4):
    m,k=x.shape;n=w.shape[0] if n is None else n
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _tiled[(triton.cdiv(n,bn),m)](x,w,out,m,n,k,chunk,lanes,bn,order,
                                 num_warps=warps,enable_fp_fusion=False)
    return out


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/small_tiled_arithmetic.json')
    args=parser.parse_args();strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'tiled cyclic FMA order search, all outputs, actual checkpoint matrices',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'trials':[]}
    for shape,case in cases.items():
        m,n,k=shape
        if m not in (1,3,6,12):continue
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        for chunk in (128,256,512):
            out=tiled(x,w,chunk)
            d=difference(ref,out)
            report['trials'].append({'shape':shape,'chunk':chunk,'lanes':16,'difference':d})
            print(shape,chunk,d['exact'],d['different_elements'],d['max_abs'],flush=True)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

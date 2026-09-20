"""Research-only exact FP32 cyclic GEMV arithmetic and reduction probes."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from benchmarks.compare import difference
from fast_moss.loading import strict_precision,REVISION

@triton.jit
def _gemv(X,W,Y,N:tl.constexpr,K:tl.constexpr,LANES:tl.constexpr,
          BN:tl.constexpr,UNROLL:tl.constexpr,REDUCE:tl.constexpr,ZERO:tl.constexpr):
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,LANES)
    acc=tl.full((BN,LANES),0,tl.float32)
    for block in tl.range(tl.cdiv(K,LANES),loop_unroll_factor=UNROLL):
        k=block*LANES+lane
        x=tl.load(X+k,k<K,0)
        w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
        acc=tl.fma(x[None,:],w,acc)
    if REDUCE=='sum':
        out=tl.sum(acc,1)
    else:
        for i in tl.static_range(0,tl.constexpr(LANES.bit_length()-1)):
            delta=LANES//2 >> i
            index=tl.broadcast_to(((lane+delta)%LANES)[None,:],(BN,LANES))
            acc=acc+tl.gather(acc,index,1)
        out=tl.reshape(tl.gather(acc,tl.full((BN,1),0,tl.int32),1),(BN,))
    if ZERO:
        out=tl.inline_asm_elementwise('add.rn.f32 $0, $1, 0f00000000;',
                                     constraints='=f,f',args=[out],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+n,out,n<N)


def gemv(x,w,lanes,bn=8,warps=4,unroll=1,reduce='down',zero=False,return_kernel=False):
    _,k=x.shape;n=w.shape[0]
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    kernel=_gemv[(triton.cdiv(n,bn),)](x,w,out,n,k,lanes,bn,unroll,reduce,zero,
                                      num_warps=warps,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'full-width one-row actual-weight cyclic GEMV arithmetic',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape,case in sorted(cases.items()):
        if shape[0]!=1:continue
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        record={'shape':shape,'trials':[]}
        for lanes in [8,16,32,64]:
            for reduce in ['down','sum']:
                out=gemv(x,w,lanes,reduce=reduce)
                d=difference(ref,out)
                bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
                record['trials'].append({'lanes':lanes,'reduction':reduce,'difference':d,'bits_equal':bits})
                print(shape,lanes,reduce,d['exact'],bits,d['different_elements'],flush=True)
        report['records'].append(record)
        Path('results/gemv_order.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

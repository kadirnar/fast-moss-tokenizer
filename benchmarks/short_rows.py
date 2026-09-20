"""Exact 2/4/8-row native-layout matrix search on original checkpoint operands."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph
from benchmarks.small_fixed_rows import fixed
from benchmarks.small_tiled_split import split, _parts, _reduce
from benchmarks.ffn_resources import resources
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


def run(x,w,config):
    strategy,rows,bn,warps,unroll=config
    return fixed(x,w,rows,bn,warps,unroll) if strategy=='fixed' else split(x,w,rows,bn,warps)


def compiled_resources(x,w,config):
    strategy,rows,bn,warps,unroll=config
    if strategy=='fixed':
        _,kernel=fixed(x,w,rows,bn,warps,unroll,return_kernel=True)
        return [resources(kernel)]
    m,k=x.shape;n=w.shape[0];parts=triton.cdiv(k,256)
    p=torch.empty((parts,m,n,16),device=x.device,dtype=x.dtype);out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    kernels=[_parts[(triton.cdiv(n,bn),triton.cdiv(m,rows),parts)](
        x,w,p,m,n,k,rows,bn,num_warps=warps,enable_fp_fusion=False),
        _reduce[(triton.cdiv(m*n,32),)](p,out,m*n,parts,32,num_warps=4,enable_fp_fusion=False)]
    return [resources(kernel) for kernel in kernels]


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'all captured 2/4/8-row learned matrices; exact row sharing and split reductions',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'previous_commit':'e5c974a','records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,case in sorted(cases.items()):
        m,n,k=shape
        if m not in (2,4,8):continue
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        graph=GraphedCallable(lambda z:(F.linear(z,w),),x)
        warm=do_bench_cudagraph(lambda:F.linear(x,w),rep=20)
        cold=evicted_replay(graph,flush,repeats=25);del graph
        record={'shape':shape,'native_warm_ms':warm,'native_cold':cold,'trials':[]}
        configs=[('fixed',rows,bn,warps,unroll) for rows in ([m,3] if m<8 else [4,8,3])
                 for bn in [4,8] for warps in [1,2] for unroll in [4,16]]
        if k>=1280:configs += [('split',rows,bn,warps,16) for rows in [m,4] for bn,warps in [(4,2),(8,4)]]
        configs=list(dict.fromkeys(configs))
        for config in configs:
            fn=lambda z:run(z,w,config)
            out=fn(x);d=difference(ref,out);bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
            compiled=compiled_resources(x,w,config)
            graph=GraphedCallable(lambda z:(fn(z),),x)
            ms=do_bench_cudagraph(lambda:fn(x),rep=15)
            cold=evicted_replay(graph,flush,repeats=25);del graph
            record['trials'].append({'config':config,'difference':d,'bits_equal':bits,
                                    'resources':compiled,'warm_ms':ms,'cold':cold})
        report['records'].append(record)
        best=min(record['trials'],key=lambda t:t['warm_ms'])
        print(shape,'native',warm,record['native_cold']['gpu_ms_median'],
              'best',best['config'],best['warm_ms'],best['cold']['gpu_ms_median'],flush=True)
        Path('results/short_rows.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

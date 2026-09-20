"""Retune native-layout SIMT grids with one/two output columns per block."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph
from fast_moss import small_matrices as sm
from fast_moss.loading import strict_precision,REVISION
from fast_moss.graphs import GraphedCallable
from benchmarks.ffn_resources import resources
from benchmarks.matrices import evicted_replay


def run(x,w,config,return_resources=False):
    m,k=x.shape;n=w.shape[0];strategy,bm,bn,warps,u=config
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    if strategy=='gemv':
        kernels=[sm._small_gemv[(triton.cdiv(n,bn),)](x,w,out,n,k,bm,bn,u,num_warps=warps,enable_fp_fusion=False)]
    elif strategy=='fixed':
        kernels=[sm._small_fixed[(triton.cdiv(n,bn),triton.cdiv(m,bm))](x,w,out,m,n,k,bm,bn,u,num_warps=warps,enable_fp_fusion=False)]
    else:
        parts=triton.cdiv(k,256);p=torch.empty((parts,m,n,16),device=x.device,dtype=x.dtype)
        kernels=[sm._small_parts[(triton.cdiv(n,bn),triton.cdiv(m,bm),parts)](x,w,p,m,n,k,bm,bn,num_warps=warps,enable_fp_fusion=False),
                 sm._small_reduce[(triton.cdiv(m*n,32),)](p,out,m*n,parts,32,num_warps=4,enable_fp_fusion=False)]
    return (out,[resources(k) for k in kernels]) if return_resources else out


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    baseline=json.loads(Path('results/narrow_baseline.json').read_text())
    report={'scope':'narrow-column native-layout grid search against supported runtime',
            'revision':REVISION,'previous_commit':'fd559c3','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for entry in baseline['small_configs']:
        shape=tuple(entry['shape']);old=tuple(entry['config']);m,n,k=shape
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        if m==1:
            configs=[('gemv',old[1],bn,1,u) for bn in [1,2] for u in [4,16,32]]
        else:
            rows=sorted(set([min(m,2),min(m,3),min(m,4),min(m,8)]))
            configs=[('fixed',r,bn,1,u) for r in rows for bn in [1,2] for u in [4,16]]
        configs=list(dict.fromkeys([old]+configs))
        row={'shape':shape,'previous_config':old,'trials':[]}
        for cfg in configs:
            fn=lambda z:run(z,w,cfg)
            out,res=run(x,w,cfg,return_resources=True)
            exact=torch.equal(out.view(torch.int32),ref.view(torch.int32))
            graph=GraphedCallable(lambda z:(fn(z),),x)
            warm=do_bench_cudagraph(lambda:fn(x),rep=20)
            cold=evicted_replay(graph,flush,repeats=25);del graph
            row['trials'].append({'config':cfg,'bits_equal':exact,'resources':res,'warm_ms':warm,'cold':cold})
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['warm_ms'])
        print(shape,'previous',row['trials'][0]['warm_ms'],'best',best['config'],best['warm_ms'],flush=True)
        Path('results/narrow_rows.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

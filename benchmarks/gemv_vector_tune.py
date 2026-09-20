"""Captured-weight CUDA vector/prefetch GEMV search against supported kernels."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_vector import gemv
from benchmarks.matrices import evicted_replay
from fast_moss.loading import strict_precision,REVISION
from fast_moss.small_matrices import CONFIGS,linear
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'exact native-layout CUDA vector-load/prefetch GEMV search',
            'previous_commit':'936b0e4','revision':REVISION,'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,old in CONFIGS.items():
        if shape[0]!=1:continue
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w);lanes=old[1]
        row={'shape':shape,'current_config':old,'controls':{},'trials':[]}
        for name,fn in [('native',lambda:F.linear(x,w)),('current',lambda:linear(x,w))]:
            graph=GraphedCallable(lambda z:(fn(),),x)
            row['controls'][name]={'warm_ms':do_bench_cudagraph(fn,rep=20),'cold':evicted_replay(graph,flush,repeats=25)};del graph
        configs=[(lanes,v,t,1,u) for v in [1,2,4,8] for t in [32,64,128] for u in [4,16]]
        configs += [(lanes,v,t,p,u) for v in [1,2,4,8] for t in [32,64] for p in [2,4] for u in [1,4]]
        for cfg in configs:
            out,res=gemv(x,w,cfg,True);bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
            graph=GraphedCallable(lambda z:(gemv(z,w,cfg),),x)
            row['trials'].append({'config':cfg,'bits_equal':bits,'resources':res,
                'warm_ms':do_bench_cudagraph(lambda:gemv(x,w,cfg),rep=15),'cold':evicted_replay(graph,flush,repeats=25)})
            del graph
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['warm_ms']);print(shape,'current',row['controls']['current']['warm_ms'],'best',best['config'],best['warm_ms'],'exact',all(t['bits_equal'] for t in row['trials']),flush=True)
        Path('results/gemv_vector_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

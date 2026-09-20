"""Search output-group storage without changing native FP32 GEMV arithmetic."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_interleaved import pack,unpack,gemv
from benchmarks.matrices import evicted_replay
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION
from fast_moss.small_matrices import CONFIGS,linear


@torch.inference_mode()
def main():
    strict_precision();inputs=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only interleaved FP32 GEMV storage; packing and extra allocation excluded from component timing',
        'previous_commit':'dafaa46','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,cfg in CONFIGS.items():
        if shape[0]!=1:continue
        x,w=inputs[shape]['x'],inputs[shape]['weight'];ref=F.linear(x,w)
        record={'shape':shape,'current_config':cfg,'controls':{},'trials':[]}
        for name,fn in [('native',lambda z:F.linear(z,w)),('current',lambda z:linear(z,w))]:
            graph=GraphedCallable(lambda z:(fn(z),),x)
            record['controls'][name]={'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20),
                                     'cold':evicted_replay(graph,flush,repeats=25)}
            del graph
        for group in (1,2,4,8,16,32):
            p=pack(w,cfg[1],group)
            assert torch.equal(unpack(p).view(torch.int32),w.view(torch.int32))
            for warps in (1,2,4):
                for unroll in (4,16,32):
                    out,res=gemv(x,p,warps,unroll,True)
                    exact=torch.equal(out.view(torch.int32),ref.view(torch.int32))
                    graph=GraphedCallable(lambda z:(gemv(z,p,warps,unroll),),x)
                    record['trials'].append({'config':(cfg[1],group,warps,unroll),'bits_equal':exact,
                        'resources':res,'packed_bytes':p.data.numel()*4,
                        'warm_ms':do_bench_cudagraph(lambda:gemv(x,p,warps,unroll),rep=15),
                        'cold':evicted_replay(graph,flush,repeats=25)})
                    del graph
            del p
        report['records'].append(record)
        best=min(record['trials'],key=lambda t:t['warm_ms'])
        print(shape,'current',record['controls']['current']['warm_ms'],'best',best['config'],best['warm_ms'],
              'exact',all(t['bits_equal'] for t in record['trials']),flush=True)
        Path('results/gemv_interleaved_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

"""Search unpadded row kernels across every captured small-N matrix shape."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.small_fixed_rows import fixed
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from benchmarks.small_fixed_baseline import previous_runtime
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION
from fast_moss.small_matrices import linear
import fast_moss.small_matrices as small

@torch.inference_mode()
@previous_runtime()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'all captured 3/6/12-row learned matrices; warm and evicted configuration search',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'previous_commit':'3cec6b1','records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,case in sorted(cases.items()):
        m,n,k=shape
        if m not in (3,6,12):continue
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        previous=lambda z:linear(z,w) if shape in small.CONFIGS else F.linear(z,w)
        graph=GraphedCallable(lambda z:(previous(z),),x)
        warm=do_bench_cudagraph(lambda:previous(x),rep=20)
        cold=evicted_replay(graph,flush,repeats=25);del graph
        record={'shape':shape,'previous_backend':small.CONFIGS.get(shape,'native'),
                'previous_warm_ms':warm,'previous_cold':cold,'trials':[]}
        configs=[(3,4,1,4),(3,4,1,16),(3,8,2,4),(3,8,2,16)] if m==3 else [
            (3,4,1,16),(6,4,1,4),(6,4,1,16),(6,8,2,16),(8,4,2,16),(6,16,4,16)]
        for config in configs:
            fn=lambda z:fixed(z,w,*config)
            out=fn(x);d=difference(ref,out)
            graph=GraphedCallable(lambda z:(fn(z),),x)
            ms=do_bench_cudagraph(lambda:fn(x),rep=20)
            cold=evicted_replay(graph,flush,repeats=25)
            record['trials'].append({'config':config,'difference':d,'warm_ms':ms,'cold':cold})
            del graph
        report['records'].append(record)
        best=min(record['trials'],key=lambda t:t['warm_ms'])
        print(shape,'previous',warm,record['previous_cold']['gpu_ms_median'],
              'best warm',best['config'],best['warm_ms'],best['cold']['gpu_ms_median'],flush=True)
        Path('results/small_fixed_shapes.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

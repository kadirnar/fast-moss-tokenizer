"""Inspect native launch geometry and search remaining ordered partitions."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from triton.testing import do_bench_cudagraph
from benchmarks.ordered_mm import tiled
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION


SHAPES=[(24,1280,1280),(48,3072,768),(48,768,768),(192,3072,768),(192,2304,768)]


@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'native launch geometry and bounded exact partition search',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape in SHAPES:
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        for _ in range(5):F.linear(x,w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(10):ref=F.linear(x,w)
            torch.cuda.synchronize()
        trace=Path('/tmp/moss-native-matrix-trace.json')
        prof.export_chrome_trace(str(trace))
        events=json.loads(trace.read_text())['traceEvents']
        kernels=[{'name':e['name'],'duration_us':e['dur'],'args':e.get('args',{})}
                 for e in events if e.get('cat')=='kernel']
        r={'shape':shape,'profile_calls':10,'kernels':kernels,'trials':[],'tiles':[]}
        split_counts=sorted({e['args']['grid'][2] for e in kernels
                             if 'cutlass::Kernel2' in e['name'] and 'grid' in e['args']})
        r['launch_partition_hints']=[{'grid_z':splits,
            'ceil_k_per_split_aligned_32':((shape[2]+splits-1)//splits+31)//32*32}
            for splits in split_counts]
        print(shape,sorted({(e['name'],str(e['args'].get('grid'))) for e in kernels}),flush=True)
        good=[]
        for chunk in range(32,shape[-1]+1,32):
            d=difference(ref,tiled(x,packed,chunk,(32,64,32,4)))
            r['trials'].append({'chunk':chunk,'difference':d})
            if d['exact']:
                good.append(chunk)
                print('exact',shape,chunk,flush=True)
        for chunk in good:
            for bm,bn in [(16,64),(32,64),(32,128),(64,64),(64,128)]:
                tile=(bm,bn,32,4)
                fn=lambda z:tiled(z,packed,chunk,tile)
                out,kernel=tiled(x,packed,chunk,tile,return_kernel=True)
                graph=GraphedCallable(lambda z:(fn(z),),x)
                r['tiles'].append({'chunk':chunk,'tile':tile,'difference':difference(ref,out),
                    'graph':difference(ref,graph(x)[0]),'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20),
                    'cold':evicted_replay(graph,flush,repeats=20),
                    'resources':{'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared}})
                del graph
        report['records'].append(r)
        Path('results/ordered_remaining.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

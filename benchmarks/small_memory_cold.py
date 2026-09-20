"""Evicted-cache search across every previously compiled small-matrix candidate."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.small_fixed_rows import fixed
from benchmarks.small_vector_cuda import vector
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from triton.testing import do_bench_cudagraph
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--warm-ms',type=int,default=20)
    parser.add_argument('--output',default='results/small_memory_cold_warm.json')
    args=parser.parse_args()
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    searches={name:json.loads(Path(f'results/{file}.json').read_text())
              for name,file in [('fixed','small_fixed_rows'),('cuda','small_vector_cuda')]}
    report={'scope':'all exact warm-search candidates, evicted-cache component search',
            'sustained_warmup_ms':args.warm_ms,'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    shapes=[tuple(r['shape']) for r in searches['fixed']['records']]
    for shape in shapes:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        graph=GraphedCallable(lambda z:(F.linear(z,w),),x)
        native_warm=do_bench_cudagraph(lambda:F.linear(x,w),rep=args.warm_ms) if args.warm_ms else None
        record={'shape':shape,'native_warm_ms':native_warm,'native':evicted_replay(graph,flush,repeats=35),'trials':[]};del graph
        for name,search in searches.items():
            row=next(r for r in search['records'] if tuple(r['shape'])==shape)
            for trial in row['trials']:
                if not trial['difference']['exact']:continue
                config=trial['config']
                fn=lambda z:(fixed(z,w,*config) if name=='fixed' else vector(z,w,tuple(config)))
                graph=GraphedCallable(lambda z:(fn(z),),x)
                d=difference(ref,graph(x)[0])
                warm=do_bench_cudagraph(lambda:fn(x),rep=args.warm_ms) if args.warm_ms else None
                cold=evicted_replay(graph,flush,repeats=25)
                record['trials'].append({'backend':name,'config':config,'difference':d,
                                        'prior_warm_ms':trial['ms'],'warm_ms':warm,'cold':cold})
                del graph
        report['records'].append(record)
        print(shape,'native',record['native']['gpu_ms_median'],'best',[(t['backend'],t['config'],t['warm_ms'],t['cold']['gpu_ms_median']) for t in sorted(record['trials'],key=lambda t:t['cold']['gpu_ms_median'])[:3]],flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

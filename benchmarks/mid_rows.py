"""Tune exact native-layout and packed matrices for 16/32/64 input rows."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.short_rows import run as small,compiled_resources
from benchmarks.ordered_shapes_confirm import linear as ordered
from benchmarks.ffn_resources import resources
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


def run(x,w,packed,config,*,exact_zero=True):
    return ordered(x,packed,config[1:],exact_zero=exact_zero) if config[0]=='ordered' else small(x,w,config)


def compiled(x,w,packed,config,*,exact_zero=True):
    if config[0]!='ordered':return compiled_resources(x,w,config)
    _,kernel=ordered(x,packed,config[1:],return_kernel=True,exact_zero=exact_zero)
    return [resources(kernel)]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy',action='store_true');parser.add_argument('--output',default='results/mid_rows.json');args=parser.parse_args()
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    arithmetic=json.load(open('results/mid_arithmetic.json'))
    report={'scope':'16/32/64-row exact partition/layout/pipeline search',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'previous_commit':'31b95eb','signed_zero_policy':'legacy' if args.legacy else 'corrected','records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for entry in arithmetic['records']:
        shape=tuple(entry['shape']);m,n,k=shape
        if not entry['exact_chunks']:continue
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous();ref=F.linear(x,w)
        graph=GraphedCallable(lambda z:(F.linear(z,w),),x)
        warm=do_bench_cudagraph(lambda:F.linear(x,w),rep=20)
        cold=evicted_replay(graph,flush,repeats=25);del graph
        row={'shape':shape,'family':entry['family'],'native_warm_ms':warm,'native_cold':cold,'trials':[]}
        if entry['family']=='small':
            configs=[('fixed',r,bn,warps,u) for r in [3,4,8] for bn in [4,8] for warps in [1,2] for u in [4,16]]
            if k>=1280:configs += [('split',r,bn,warps,16) for r in [4,8] for bn,warps in [(4,2),(8,4)]]
        else:
            configs=[('ordered',chunk,bm,bn,32,4,stage) for chunk in entry['exact_chunks']
                     for bm,bn in [(16,64),(32,64),(32,128),(64,64),(64,128)] for stage in [2,3]]
        for config in configs:
            fn=lambda z:run(z,w,packed,config,exact_zero=not args.legacy)
            out=fn(x);d=difference(ref,out);bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
            res=compiled(x,w,packed,config,exact_zero=not args.legacy)
            graph=GraphedCallable(lambda z:(fn(z),),x)
            ms=do_bench_cudagraph(lambda:fn(x),rep=20);cold=evicted_replay(graph,flush,repeats=25);del graph
            row['trials'].append({'config':config,'difference':d,'bits_equal':bits,'resources':res,'warm_ms':ms,'cold':cold})
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['warm_ms'])
        print(shape,'native',warm,row['native_cold']['gpu_ms_median'],'best',best['config'],best['warm_ms'],best['cold']['gpu_ms_median'],flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

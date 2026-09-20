"""Actual checkpoint matrix gates for multi-pass tensor-core arithmetic."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.tensorcore_mm import linear
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.loading import strict_precision,REVISION
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--rows',type=int,nargs='+',default=[24,128,384,3072])
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--output',default='results/tensorcore_shapes.json')
    p.add_argument('--compact',action='store_true')
    p.add_argument('--mode',choices=['tf32x3','bf16x6'],default='tf32x3')
    a=p.parse_args();strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    chosen=sorted(((s,c) for s,c in cases.items() if s[0] in a.rows),key=lambda x:x[1]['calls']*x[0][1]*x[0][2],reverse=True)
    if a.limit:chosen=chosen[:a.limit]
    report={'scope':'experimental actual-weight multi-pass tensor-core GEMM; not promoted',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'storage_dtype':'float32','torch_tf32':False,'candidate_arithmetic':a.mode,
            'records':[],'failures':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,case in chosen:
        x,w=case['x'],case['weight'];gold=x.double()@w.double().T;ref=F.linear(x,w)
        configs=[None,(32,64,32,4,3),(64,64,32,4,3),(64,128,32,4,3),(32,128,32,4,3),(32,64,64,4,3)]
        if a.compact:configs=[None,(16,32,32,4,2),(16,64,32,4,2),(32,32,32,4,2),
                             (32,64,32,8,2),(64,64,32,8,2),(64,128,32,8,2),(32,64,16,4,2)]
        for config in configs:
            fn=(lambda z:F.linear(z,w)) if config is None else (lambda z:linear(z,w,config,mode=a.mode))
            try:
                out=fn(x);graph=GraphedCallable(lambda z:(fn(z),),x)
                r={'shape_MNK':shape,'layer':case['name'],'tile':config,'reference':difference(ref,out),
                   'graph_vs_eager':difference(out,graph(x)[0]),'fp64_max_abs':(out.double()-gold).abs().max().item(),
                   'fp64_rmse':(out.double()-gold).square().mean().sqrt().item(),
                   'hot_graph_ms':do_bench_cudagraph(lambda:fn(x),rep=20,return_mode='median'),
                   'evicted_replay':evicted_replay(graph,flush)}
                if config is not None:
                    _,kernel=linear(x,w,config,return_kernel=True,mode=a.mode)
                    r['resources']={'registers':kernel.n_regs,'spills':kernel.n_spills,
                                    'mma_instructions_in_ptx':kernel.asm['ptx'].count('mma.sync')}
                report['records'].append(r)
                print(shape,config,'us',round(r['hot_graph_ms']*1000,2),'maxerr',r['reference']['max_abs'],flush=True)
                del graph
            except Exception as error:
                report['failures'].append({'shape':shape,'tile':config,'error':str(error)});print('failure',shape,config,error,flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

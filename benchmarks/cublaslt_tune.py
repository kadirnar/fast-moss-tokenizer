"""Bounded cuBLASLt configuration search with exactness and cold-cache gates."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.cublaslt_search import candidates
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',default='results/matrix_inputs.pt')
    p.add_argument('--rows',type=int,nargs='+',default=[1,3,24,384])
    p.add_argument('--limit',type=int,default=8)
    p.add_argument('--layouts',nargs='+',default=['col','packed'])
    p.add_argument('--split-k',type=int,nargs='+',default=[0,1,2,3,4,5,6,7,8,10,12,16,20,24,32])
    p.add_argument('--output',default='results/cublaslt_config_search.json')
    a=p.parse_args();strict_precision()
    cases=torch.load(a.inputs,weights_only=True)
    ordered=sorted(((k,v) for k,v in cases.items() if k[0] in a.rows),key=lambda kv:kv[1]['calls']*kv[0][1]*kv[0][2],reverse=True)
    if a.limit:ordered=ordered[:a.limit]
    report={'scope':'bounded capability-based FP32 matrix search, not model speedup',
            'revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cublaslt_version':library().cublasLtGetVersion(),'dtype':'float32','tf32':False,
            'compute':'CUBLAS_COMPUTE_32F_PEDANTIC','reference_layout':'flattened contiguous 2-D',
            'custom_search':'all values up to 127; otherwise 0..31, individual bits, low-bit masks, maximum',
            'split_k':a.split_k,'heuristic_seeds_and_split_factors_included':True,
            'records':[],'trials':[],'failures':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    def record(shape,case,name,fn,algorithm=None,config=None):
        x,w=case['x'],case['weight'];ref=F.linear(x,w);out=fn(x)
        graph=GraphedCallable(lambda z:(fn(z),),x)
        r={'shape_MNK':shape,'layer':case['name'],'backend':name,'algorithm':algorithm,'configuration':config,
           'reference_difference':difference(ref,out),'graph_vs_eager':difference(out,graph(x)[0]),
           'hot_graph_ms':do_bench_cudagraph(lambda:fn(x),rep=20,return_mode='median'),
           'evicted_replay':evicted_replay(graph,flush)}
        del graph
        return r
    for shape,case in ordered:
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        baseline=record(shape,case,'torch',lambda z:F.linear(z,w))
        report['records'].append(baseline)
        for layout in a.layouts:
            with LinearPlan(w,shape[0],layout) as plan:
                timings=[];trials=[]
                for result,config in candidates(plan,a.split_k):
                    index=len(plan.algorithms);plan.algorithms.append(result)
                    trial={'shape_MNK':shape,'layout':layout,'configuration':config}
                    try:
                        out=plan(x,index)
                        trial['exact']=torch.equal(ref,out)
                        if trial['exact']:
                            ms=do_bench_cudagraph(lambda:plan(x,index),rep=5,return_mode='median')
                            trial['hot_graph_ms']=ms;timings.append((ms,index,config))
                    except RuntimeError as error:
                        trial['error']=str(error);report['failures'].append(trial)
                    trials.append(trial)
                report['trials'].extend(trials)
                print(shape,layout,'tested',len(trials),'exact',len(timings),flush=True)
                for _,index,config in sorted(timings)[:8]:
                    r=record(shape,case,f'lt_{layout}_{index}',lambda z:plan(z,index),plan.metadata(index),config)
                    report['records'].append(r)
                    print(' finalist',config,'warm_gain',baseline['hot_graph_ms']/r['hot_graph_ms'],
                          'cold_gain',baseline['evicted_replay']['gpu_ms_median']/r['evicted_replay']['gpu_ms_median'],flush=True)
            Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

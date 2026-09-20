"""Exact GEMV epilogues: launch tuning, edge cases, warm and distinct-weight rings."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_epilogue import linear as fused
from fast_moss.small_matrices import CONFIGS,linear
from fast_moss.kernels import scale_add
from fast_moss.ffn import math_library
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(12769);library=math_library()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only exact native GEMV epilogues; component graph timing excludes graph copies and setup',
        'previous_commit':'21cb936','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'seed':12769,'records':[]}
    for shape,mode in [((1,5120,1280),'gelu'),((1,1280,5120),'residual')]:
        x,w=cases[shape]['x'],cases[shape]['weight'];residual=torch.randn(1,shape[1],device='cuda');scale=torch.randn(shape[1],device='cuda')*.01
        def epilogue(y):return F.gelu(y) if mode=='gelu' else residual+y*scale
        def current(z,weight):
            y=linear(z,weight)
            return F.gelu(y) if mode=='gelu' else scale_add(residual,y,scale)
        variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
            ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
            ('large',torch.randn_like(x)*1e20),('negative_minimum',torch.full_like(x,-1.401298464324817e-45)),
            ('negative_zero',torch.full_like(x,-0.))]
        variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
        sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
        refs=[epilogue(F.linear(z,w)).view(torch.int32) for _,z in variants]
        record={'shape':shape,'mode':mode,'current_config':CONFIGS[shape],
            'current_ms':do_bench_cudagraph(lambda:current(x,w),rep=30),'trials':[]}
        lanes=CONFIGS[shape][1]
        for bn in (1,2,4,8):
            for warps in (1,2):
                for u in (4,16,32):
                    cfg=('gemv',lanes,bn,warps,u)
                    fn=lambda z:fused(z,w,mode,residual,scale,library,cfg)
                    out,resources=fused(x,w,mode,residual,scale,library,cfg,True)
                    graph=GraphedCallable(lambda z:(fn(z),),x)
                    checks=[{'input':label,'eager':torch.equal(ref,fn(z).view(torch.int32)),
                            'graph':torch.equal(ref,graph(z)[0].view(torch.int32))}
                            for (label,z),ref in zip(variants,refs)]
                    del graph
                    positive=torch.full_like(w,.125);z=torch.full_like(x,-1.401298464324817e-45)
                    ref=epilogue(F.linear(z,positive)).view(torch.int32)
                    fn_positive=lambda z:fused(z,positive,mode,residual,scale,library,cfg)
                    graph=GraphedCallable(lambda z:(fn_positive(z),),z)
                    checks.append({'input':'positive_weight_underflow','eager':torch.equal(ref,fn_positive(z).view(torch.int32)),
                                   'graph':torch.equal(ref,graph(z)[0].view(torch.int32))});del graph
                    record['trials'].append({'config':cfg,'resources':resources,'checks':checks,
                        'all_exact':all(c['eager'] and c['graph'] for c in checks),
                        'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20)})
        valid=[t for t in record['trials'] if t['all_exact'] and t['resources']['spills']==0]
        configs=list(dict.fromkeys([tuple(t['config']) for t in sorted(valid,key=lambda t:t['warm_ms'])[:2]]+[CONFIGS[shape]]))
        weights=[w.clone() for _ in range(32)];record['rings']=[]
        for length in (1,4,32):
            fns={'current':lambda z:tuple(current(z,t) for t in weights[:length])}
            for cfg in configs:
                fns[json.dumps(cfg)]=lambda z,cfg=cfg:tuple(fused(z,t,mode,residual,scale,library,cfg) for t in weights[:length])
            checks={};rounds=[]
            for name,fn in fns.items():
                graph=GraphedCallable(fn,x)
                checks[name]={'eager':all(torch.equal(t.view(torch.int32),refs[0]) for t in fn(x)),
                              'graph':all(torch.equal(t.view(torch.int32),refs[0]) for t in graph(x))};del graph
            for repeat in range(3):
                names=list(fns);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':do_bench_cudagraph(lambda:fns[name](x),rep=30)/length})
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in fns}
            record['rings'].append({'length':length,'checks':checks,'rounds':rounds,'medians_ms':medians,
                                   'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}})
            print(shape,mode,length,record['rings'][-1]['speedups'],flush=True)
        report['records'].append(record)
        Path('results/gemv_epilogue_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    if not all(t['all_exact'] for r in report['records'] for t in r['trials']):raise SystemExit('Fidelity mismatch')

if __name__=='__main__':main()

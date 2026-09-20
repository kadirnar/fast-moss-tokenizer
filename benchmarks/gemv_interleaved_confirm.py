"""Stress and repeat the best exact interleaved-storage GEMV schedules."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_interleaved import pack,gemv
from benchmarks.matrices import evicted_replay
from fast_moss.small_matrices import linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(81294)
    inputs=torch.load('results/matrix_inputs.pt',weights_only=True)
    search=json.loads(Path('results/gemv_interleaved_tune.json').read_text())
    report={'scope':'interleaved FP32 GEMV: stress and three rotating warm/evicted rounds',
        'previous_commit':'dafaa46','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'seed':81294,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for row in search['records']:
        shape=tuple(row['shape']);x,w=inputs[shape]['x'],inputs[shape]['weight']
        valid=[t for t in row['trials'] if t['bits_equal'] and t['resources']['spills']==0]
        best=sorted(valid,key=lambda t:t['warm_ms'])[:2]+sorted(valid,key=lambda t:t['cold']['gpu_ms_median'])[:2]
        configs=list(dict.fromkeys(tuple(t['config']) for t in best))
        fns={'native':lambda z:F.linear(z,w),'current':lambda z:linear(z,w)}
        resources={};packed={}
        for cfg in configs:
            lanes,group,warps,unroll=cfg;name=json.dumps(cfg)
            if (lanes,group) not in packed:packed[lanes,group]=pack(w,lanes,group)
            p=packed[lanes,group]
            fns[name]=lambda z,p=p,warps=warps,unroll=unroll:gemv(z,p,warps,unroll)
            resources[name]=gemv(x,p,warps,unroll,True)[1]
        variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
            ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
            ('large',torch.randn_like(x)*1e20),('negative_minimum',torch.full_like(x,-1.401298464324817e-45)),
            ('negative_zero',torch.full_like(x,-0.))]
        variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
        sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
        checks=[]
        for name,fn in fns.items():
            graph=GraphedCallable(lambda z:(fn(z),),x)
            for label,z in variants:
                ref=F.linear(z,w).view(torch.int32)
                checks.append({'backend':name,'input':label,
                    'eager_bits_equal':torch.equal(ref,fn(z).view(torch.int32)),
                    'graph_bits_equal':torch.equal(ref,graph(z)[0].view(torch.int32))})
            del graph
            positive=torch.full_like(w,.125);z=torch.full_like(x,-1.401298464324817e-45)
            if name=='native':underflow=lambda z:F.linear(z,positive)
            elif name=='current':underflow=lambda z:linear(z,positive)
            else:
                lanes,group,warps,unroll=json.loads(name);p_pos=pack(positive,lanes,group)
                underflow=lambda z:gemv(z,p_pos,warps,unroll)
            graph=GraphedCallable(lambda z:(underflow(z),),z);ref=F.linear(z,positive).view(torch.int32)
            checks.append({'backend':name,'input':'positive_weight_underflow',
                'eager_bits_equal':torch.equal(ref,underflow(z).view(torch.int32)),
                'graph_bits_equal':torch.equal(ref,graph(z)[0].view(torch.int32))})
            del graph,underflow
        exact={name:all(c['eager_bits_equal'] and c['graph_bits_equal'] for c in checks if c['backend']==name) for name in fns}
        rounds=[]
        for repeat in range(3):
            names=list(fns);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                if not exact[name]:continue
                fn=fns[name];graph=GraphedCallable(lambda z:(fn(z),),x)
                rounds.append({'round':repeat,'backend':name,'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20),
                    'cold':evicted_replay(graph,flush,repeats=35)});del graph
        medians={name:{'warm':statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                       'cold':statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)}
                 for name in fns if exact[name]}
        report['records'].append({'shape':shape,'configs':configs,'resources':resources,'checks':checks,
                                  'exact':exact,'rounds':rounds,'medians_ms':medians})
        print(shape,'exact',exact,'medians',medians,flush=True)
        Path('results/gemv_interleaved_confirm.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

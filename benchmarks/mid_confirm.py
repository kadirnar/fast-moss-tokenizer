"""Stress exact 16/32/64-row candidates and compare warm/evicted component latency."""
import argparse
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.mid_rows import run,compiled
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--legacy',action='store_true');parser.add_argument('--output',default='results/mid_confirm.json');args=parser.parse_args()
    strict_precision();torch.manual_seed(63724)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    search=json.loads(Path('results/mid_rows.json').read_text())
    report={'scope':'16/32/64-row stress and three alternating warm/evicted rounds',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':63724,'signed_zero_policy':'legacy' if args.legacy else 'corrected','records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for record in search['records']:
        shape=tuple(record['shape']);x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        valid=[t for t in record['trials'] if t['bits_equal'] and all(k['spills']==0 for k in t['resources'])
               and t['warm_ms']<record['native_warm_ms']*.99]
        if not valid:continue
        best=sorted(valid,key=lambda t:t['warm_ms'])[:2]+[min(valid,key=lambda t:t['cold']['gpu_ms_median'])]
        configs=list(dict.fromkeys(tuple(t['config']) for t in best))
        fns={'native':lambda z:F.linear(z,w)}
        resources={}
        for config in configs:
            name=json.dumps(config)
            fns[name]=lambda z,cfg=config:run(z,w,packed,cfg,exact_zero=not args.legacy)
            resources[name]=compiled(x,w,packed,config,exact_zero=not args.legacy)
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
                ref=F.linear(z,w);out=fn(z);replay=graph(z)[0]
                checks.append({'backend':name,'input':label,'eager':difference(ref,out),'graph':difference(ref,replay),
                    'eager_bits_equal':torch.equal(ref.view(torch.int32),out.view(torch.int32)),
                    'graph_bits_equal':torch.equal(ref.view(torch.int32),replay.view(torch.int32))})
            del graph
        exact={name:all(c['eager_bits_equal'] and c['graph_bits_equal'] for c in checks if c['backend']==name) for name in fns}
        rounds=[]
        for repeat in range(3):
            names=list(fns)
            if repeat%2:names.reverse()
            for name in names:
                if not exact[name]:continue
                fn=fns[name];graph=GraphedCallable(lambda z:(fn(z),),x)
                rounds.append({'round':repeat,'backend':name,'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20),
                               'cold':evicted_replay(graph,flush,repeats=35)})
                del graph
        medians={name:{'warm':statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                       'cold':statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)}
                       for name in fns if exact[name]}
        result={'shape':shape,'resources':resources,'configs':configs,'checks':checks,'exact':exact,'rounds':rounds,'medians_ms':medians}
        report['records'].append(result)
        print(shape,'exact',exact,'medians',medians,flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

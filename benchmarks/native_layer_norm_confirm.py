"""Native FP32 LayerNorm finalists: input/statistic stress and rotating timings."""
import argparse
import json
import statistics
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from benchmarks.native_layer_norm import layer_norm
from benchmarks.native_layer_norm_tune import bits
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='results/native_layer_norm_tune.json')
    parser.add_argument('--output',default='results/native_layer_norm_confirm.json');args=parser.parse_args()
    strict_precision();torch.manual_seed(57139)
    cases=torch.load('results/layer_norm_inputs.pt',weights_only=True)
    search=json.load(open(args.input))
    report={'scope':'actual LayerNorm finalist validation; outputs, means and reciprocal stddev checked bitwise',
        'previous_commit':'52a2c91','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'records':[]}
    for row in search['records']:
        shape=tuple(row['shape']);r=cases[shape];x,g,b,eps=(r[k] for k in ('x','weight','bias','eps'))
        valid=[t for t in row['trials'] if all(t['bits_equal']) and t['resources']['local_bytes']==0]
        configs=list(dict.fromkeys(tuple(t['config']) for t in sorted(valid,key=lambda t:t['ms'])[:2]))
        fns={'native':lambda z:torch.native_layer_norm(z,(shape[1],),g,b,eps)}
        for cfg in configs:fns[json.dumps(cfg)]=lambda z,cfg=cfg:layer_norm(z,g,b,eps,cfg)
        variants=[('actual',x),('zero',torch.zeros_like(x)),('negative_zero',torch.full_like(x,-0.)),
            ('minimum',torch.full_like(x,-1.401298464324817e-45)),('subnormal',torch.randn_like(x)*1e-38),
            ('tiny',torch.randn_like(x)*1e-20),('large',torch.randn_like(x)*1e20),
            ('offset',torch.randn_like(x)*.01+1e4),('constant',torch.full_like(x,.125)),
            ('negated',-x),('random',torch.randn_like(x)),('scaled',x*.17)]
        checks=[]
        for name,fn in fns.items():
            graph=GraphedCallable(fn,x)
            for label,z in variants:
                ref=fns['native'](z);out=fn(z);replay=graph(z)
                checks.append({'backend':name,'input':label,'eager':[bits(a,v) for a,v in zip(ref,out)],
                               'graph':[bits(a,v) for a,v in zip(ref,replay)]})
            del graph
        exact={name:all(all(c['eager']) and all(c['graph']) for c in checks if c['backend']==name) for name in fns}
        rounds=[]
        for repeat in range(3):
            names=list(fns);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                if not exact[name]:continue
                rounds.append({'round':repeat,'backend':name,'ms':do_bench_cudagraph(lambda:fns[name](x),rep=30)})
        medians={name:statistics.median(r['ms'] for r in rounds if r['backend']==name) for name in fns if exact[name]}
        report['records'].append({'shape':shape,'configs':configs,'checks':checks,'exact':exact,'rounds':rounds,'medians_ms':medians})
        print(shape,exact,medians,flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

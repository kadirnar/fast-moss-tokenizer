"""Stress exact small-row candidates and compare warm/evicted component latency."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.small_tiled_grouped import grouped
from benchmarks.small_tiled_split import split
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(97651)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    searches={name:json.loads(Path(f'results/small_tiled_{name}.json').read_text())
              for name in ['grouped','split']}
    selected={}
    for name,search in searches.items():
        for r in search['records']:
            candidates=sorted([t for t in r['trials'] if t['difference']['exact']],key=lambda t:t['ms'])[:2]
            selected.setdefault(tuple(r['shape']),[]).extend((name,t['config']) for t in candidates)
    report={'scope':'small-row arithmetic stress and three alternating warm/evicted component rounds',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'seed':97651,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,configs in selected.items():
        x,w=cases[shape]['x'],cases[shape]['weight']
        fns={'native':lambda z:F.linear(z,w)}
        for name,config in configs:
            fns[f'{name}_{config}']=lambda z,name=name,config=config:({'grouped':grouped,'split':split}[name])(z,w,*config)
        variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
                  ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
                  ('large',torch.randn_like(x)*1e20)]
        variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
        sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
        checks=[]
        for name,fn in fns.items():
            graph=GraphedCallable(lambda z:(fn(z),),x)
            for label,z in variants:
                ref=F.linear(z,w);out=fn(z);replay=graph(z)[0]
                checks.append({'backend':name,'input':label,'eager':difference(ref,out),
                    'graph':difference(ref,replay),'bits_equal':torch.equal(ref.view(torch.int32),out.view(torch.int32))})
            del graph
        exact={name:all(c['eager']['exact'] and c['graph']['exact'] and c['bits_equal']
                          for c in checks if c['backend']==name) for name in fns}
        rounds=[]
        for repeat in range(3):
            names=list(fns)
            if repeat%2:names.reverse()
            for name in names:
                if not exact[name]:continue
                fn=fns[name];graph=GraphedCallable(lambda z:(fn(z),),x)
                rounds.append({'round':repeat,'backend':name,'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=20),
                               'cold':evicted_replay(graph,flush,repeats=25)})
                del graph
        medians={name:{'warm':statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                       'cold':statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)}
                       for name in fns if exact[name]}
        record={'shape':shape,'configs':configs,'checks':checks,'exact':exact,'rounds':rounds,'medians_ms':medians}
        report['records'].append(record)
        print(shape,'exact',exact,'medians',medians,flush=True)
        Path('results/small_tiled_confirm.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

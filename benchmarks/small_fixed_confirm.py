"""Stress exact small-row candidates and compare warm/evicted component latency."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.small_fixed_rows import fixed
from fast_moss.small_matrices import linear
import fast_moss.small_matrices as small
from benchmarks.ffn_resources import resources
from benchmarks.compare import difference
from benchmarks.small_fixed_baseline import previous_runtime
from benchmarks.matrices import evicted_replay
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION

@torch.inference_mode()
@previous_runtime()
def main():
    strict_precision();torch.manual_seed(97651)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    search=json.loads(Path('results/small_fixed_shapes.json').read_text())
    selected={}
    for record in search['records']:
        valid=[t for t in record['trials'] if t['difference']['exact'] and t['warm_ms']<record['previous_warm_ms']*.99]
        if not valid:continue
        picks=[min(valid,key=lambda t:t['warm_ms']),min(valid,key=lambda t:t['cold']['gpu_ms_median'])]
        selected[tuple(record['shape'])]=list(dict.fromkeys(tuple(t['config']) for t in picks))
    report={'scope':'all small fixed-row candidates; stress and three alternating warm/evicted rounds against committed runtime',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'seed':97651,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,configs in selected.items():
        x,w=cases[shape]['x'],cases[shape]['weight']
        fns={'previous':lambda z:linear(z,w) if shape in small.CONFIGS else F.linear(z,w)}
        compiled={}
        for config in configs:
            name=str(list(config))
            fns[name]=lambda z,config=config:fixed(z,w,*config)
            _,kernel=fixed(x,w,*config,return_kernel=True)
            compiled[name]=resources(kernel)
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
        record={'shape':shape,'resources':compiled,'configs':configs,'checks':checks,'exact':exact,'rounds':rounds,'medians_ms':medians}
        report['records'].append(record)
        print(shape,'exact',exact,'medians',medians,flush=True)
        Path('results/small_fixed_confirm.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

"""Exactness stress and distinct-weight rings for long-K shared-memory partials."""
import hashlib
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.wide_cta_matrix import linear,SOURCE
from fast_moss.small_matrices import linear as current
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision,REVISION
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(9153);cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    sweep=json.loads(Path('results/wide_cta_tune.json').read_text())
    report={'scope':'actual-weight stress and synthetic distinct-allocation rings; copies/capture excluded; no physical DRAM counters or codec performance claim',
        'previous_commit':'397db43','source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for row in sweep['records']:
        shape=tuple(row['shape']);x,w=cases[shape]['x'],cases[shape]['weight']
        trials=sorted((t for t in row['trials'] if t['bits_equal'] and t['resources']['local_bytes']==0),key=lambda t:t['ms'])
        # Confirm two fast candidates with distinct parallel partition counts.
        configs=[]
        for trial in trials:
            if not any(c[2]==trial['config'][2] for c in configs):configs.append(tuple(trial['config']))
            if len(configs)==2:break
        methods={'native':lambda a,b:F.linear(a,b),'current':current}
        methods.update({json.dumps(cfg):lambda a,b,cfg=cfg:linear(a,b,cfg) for cfg in configs})
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal',x*1e-38,w),('tiny',x*1e-20,w),('large',x*1e20,w),('random',torch.randn_like(x),w),
            ('sparse',x*(torch.arange(x.shape[1],device=x.device)%29==0),w),
            ('positive_weight_underflow',torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]
        checks=[]
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,b:(fn(a,b),),x,w)
            for label,a,b in variants:
                reference=F.linear(a,b)
                checks.append({'backend':name,'input':label,'eager':bits(fn(a,b),reference),'graph':bits(graph(a,b)[0],reference)})
            del graph
        exact={name:all(c['eager'] and c['graph'] for c in checks if c['backend']==name) for name in methods}
        weights=[w.clone() for _ in range(32)];assert len({v.data_ptr() for v in weights})==32
        rings=[]
        for length in (1,8,32):
            functions={name:lambda fn=fn:tuple(fn(x,v) for v in weights[:length]) for name,fn in methods.items() if exact[name]}
            ring_checks={}
            reference=F.linear(x,w)
            for name,fn in functions.items():
                graph=GraphedCallable(lambda z,method=methods[name]:tuple(method(z,v) for v in weights[:length]),x)
                ring_checks[name]={'eager':all(bits(y,reference) for y in fn()),
                                  'graph':all(bits(y,reference) for y in graph(x))}
                del graph
            assert all(c['eager'] and c['graph'] for c in ring_checks.values())
            rounds=[]
            for repeat in range(3):
                names=list(functions);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:
                    ms=do_bench_cudagraph(functions[name],rep=25)
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':ms/length})
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in functions}
            rings.append({'length':length,'logical_weight_bytes':length*w.numel()*4,'checks':ring_checks,'rounds':rounds,'per_call_ms':medians,
                'speedups':{name:medians['current']/value for name,value in medians.items() if name not in ('native','current')}})
            print(shape,length,exact,rings[-1]['speedups'],flush=True)
        report['records'].append({'shape':shape,'configs':configs,'checks':checks,'exact':exact,'rings':rings})
        Path('results/wide_cta_confirm.json').write_text(json.dumps(report,indent=2)+'\n')
        del weights,functions,methods,variants

if __name__=='__main__':main()

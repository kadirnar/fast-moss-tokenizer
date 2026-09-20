"""Confirm epilogue-aware CUDA schedules with exact stress and allocation rings."""
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.short_ffn import CONFIGS,linear as fused
from benchmarks.native_layer_norm_tune import bits
from fast_moss.wide_matrices import linear as current_matrix
from fast_moss.kernels import scale_add
from fast_moss.normalization import compiler
from fast_moss.ffn import math_library
from fast_moss.loading import strict_precision,REVISION
from fast_moss.graphs import GraphedCallable

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(9331);bindings=compiler();library=math_library();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    sweep=json.loads(Path('results/short_ffn_retune.json').read_text());report={'scope':'epilogue-aware CUDA confirmation; original/current controls and synthetic distinct-allocation rings','previous_commit':'c67e3a9','revision':REVISION,'records':[]}
    for row in sweep['records']:
        shape=tuple(row['shape']);x,w=cases[shape]['x'],cases[shape]['weight'];m,n,k=shape
        residual=torch.randn(m,n,device='cuda');scale=torch.randn(n,device='cuda')*.01
        configs=[]
        for t in sorted(row['trials'],key=lambda t:t['ms'] if t['ms'] is not None else float('inf')):
            if t['bits_equal'] and not any(c[2]==t['config'][2] for c in configs):configs.append(tuple(t['config']))
            if len(configs)==3:break
        def candidate(a,b,cfg):
            CONFIGS[shape]=('wide',cfg);return fused(a,b,'residual',residual,scale,library,bindings)
        methods={'native':lambda a,b:residual+F.linear(a,b)*scale,'current':lambda a,b:scale_add(residual,current_matrix(a,b,bindings),scale)}
        methods.update({json.dumps(c):lambda a,b,c=c:candidate(a,b,c) for c in configs})
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal',x*1e-38,w),('tiny',x*1e-20,w),('large',x*1e20,w),('random',torch.randn_like(x),w),
            ('sparse',x*(torch.arange(k,device='cuda')%29==0),w),
            ('positive_weight_underflow',torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]
        checks=[]
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,b:(fn(a,b),),x,w)
            for label,a,b in variants:
                ref=residual+F.linear(a,b)*scale;checks.append({'backend':name,'input':label,'eager':bits(fn(a,b),ref),'graph':bits(graph(a,b)[0],ref)})
            del graph
        assert all(c['eager'] and c['graph'] for c in checks)
        weights=[w.clone() for _ in range(32)];assert len({v.data_ptr() for v in weights})==32
        rings=[]
        for length in (1,8,32):
            funcs={name:lambda z,fn=fn:tuple(fn(z,b) for b in weights[:length]) for name,fn in methods.items() if name!='native'}
            ref=residual+F.linear(x,w)*scale;ring_checks={};rounds=[]
            for name,fn in funcs.items():
                graph=GraphedCallable(fn,x);ring_checks[name]={'eager':all(bits(y,ref) for y in fn(x)),'graph':all(bits(y,ref) for y in graph(x))};del graph
            assert all(c['eager'] and c['graph'] for c in ring_checks.values())
            for repeat in range(3):
                names=list(funcs);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:rounds.append({'round':repeat,'backend':name,'per_call_ms':do_bench_cudagraph(lambda:funcs[name](x),rep=30)/length})
            medians={name:statistics.median(v['per_call_ms'] for v in rounds if v['backend']==name) for name in funcs}
            rings.append({'length':length,'logical_weight_bytes':length*w.numel()*4,'checks':ring_checks,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}})
            print(shape,length,rings[-1]['speedups'],flush=True)
        report['records'].append({'shape':shape,'configs':configs,'checks':checks,'rings':rings});Path('results/short_ffn_confirm.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

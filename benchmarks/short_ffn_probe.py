"""Captured short-FFN fidelity, resources, and rotating distinct-allocation rings."""
import hashlib
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.short_ffn import CONFIGS,linear as fused,cuda_source
from fast_moss.small_matrices import linear as small
from fast_moss.cuda_matrices import linear as narrow
from fast_moss.wide_matrices import linear as wide
from fast_moss.kernels import scale_add
from fast_moss.ffn import math_library
from fast_moss.normalization import compiler
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION
from benchmarks.native_layer_norm_tune import bits

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(9771);library=math_library();bindings=compiler()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'short FFN epilogues on current matrix schedules; exact stress and distinct-allocation component rings; copies and capture excluded',
        'previous_commit':'c67e3a9','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'cuda_source_sha256':{f:hashlib.sha256(cuda_source(f).encode()).hexdigest() for f in ('cuda','wide')},'records':[]}
    for shape,(family,config) in sorted(CONFIGS.items()):
        x,w=cases[shape]['x'],cases[shape]['weight'];m,n,k=shape;mode='gelu' if n>k else 'residual'
        residual=torch.randn(m,n,device='cuda');scale=torch.randn(n,device='cuda')*.01
        def epilogue(y):return F.gelu(y) if mode=='gelu' else residual+y*scale
        def current(a,b):
            y=small(a,b) if family=='fixed' else (wide if family=='wide' else narrow)(a,b,bindings)
            return F.gelu(y) if mode=='gelu' else scale_add(residual,y,scale)
        methods={'native':lambda a,b:epilogue(F.linear(a,b)), 'current':current,
                 'fused':lambda a,b:fused(a,b,mode,residual,scale,library,bindings)}
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal',x*1e-38,w),('tiny',x*1e-20,w),('large',x*1e20,w),('random',torch.randn_like(x),w),
            ('sparse',x*(torch.arange(k,device='cuda')%29==0),w),
            ('positive_weight_underflow',torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]
        checks=[]
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,b:(fn(a,b),),x,w)
            for label,a,b in variants:
                ref=epilogue(F.linear(a,b));checks.append({'backend':name,'input':label,'eager':bits(fn(a,b),ref),'graph':bits(graph(a,b)[0],ref)})
            del graph
        out,resources=fused(x,w,mode,residual,scale,library,bindings,True)
        record={'shape':shape,'mode':mode,'family':family,'config':config,'checks':checks,'all_exact':all(c['eager'] and c['graph'] for c in checks),'resources':resources,'rings':[]}
        if not record['all_exact']:
            report['records'].append(record);Path('results/short_ffn_probe.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(f'Fidelity failed: {shape}')
        weights=[w.clone() for _ in range(32)];assert len({v.data_ptr() for v in weights})==32
        for length in (1,8,32):
            funcs={name:lambda z,fn=fn:tuple(fn(z,b) for b in weights[:length]) for name,fn in methods.items() if name!='native'}
            ref=epilogue(F.linear(x,w));ring_checks={};rounds=[]
            for name,fn in funcs.items():
                graph=GraphedCallable(fn,x);ring_checks[name]={'eager':all(bits(y,ref) for y in fn(x)),'graph':all(bits(y,ref) for y in graph(x))};del graph
            assert all(c['eager'] and c['graph'] for c in ring_checks.values())
            for repeat in range(3):
                for name in (['current','fused'] if repeat%2==0 else ['fused','current']):
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':do_bench_cudagraph(lambda:funcs[name](x),rep=30)/length})
            medians={name:statistics.median(v['per_call_ms'] for v in rounds if v['backend']==name) for name in funcs}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'checks':ring_checks,'rounds':rounds,'medians_ms':medians,'speedup':medians['current']/medians['fused']})
            print(shape,mode,family,length,record['rings'][-1]['speedup'],flush=True)
        report['records'].append(record);Path('results/short_ffn_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        del methods,variants,weights,funcs

if __name__=='__main__':main()

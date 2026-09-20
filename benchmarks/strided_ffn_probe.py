"""Strided FFN contraction: captured operands, exact output layouts, allocation rings."""
import hashlib
import json
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.strided_ffn import CONFIGS,linear as fused,cuda_source,_strided_ffn_fixed,_epilogue
from fast_moss.small_matrices import linear as small
from fast_moss.wide_matrices import linear as wide
from fast_moss.ffn import math_library
from fast_moss.normalization import compiler
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION
from benchmarks.native_layer_norm_tune import bits

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(3021);library=math_library();bindings=compiler()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'strided FFN contractions; exact stress and synthetic distinct-allocation rings, copies/capture excluded',
        'previous_commit':'531fd32','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':{'cuda':hashlib.sha256(cuda_source('wide').encode()).hexdigest(),
            'triton':hashlib.sha256(_strided_ffn_fixed.src.encode()).hexdigest(),'epilogue':hashlib.sha256(_epilogue.src.encode()).hexdigest()},'records':[]}
    output=Path('results/strided_ffn_probe.json')
    for shape,(family,config) in sorted(CONFIGS.items()):
        m,n,k=shape
        if n>=k:continue
        x,w=cases[shape]['x'],cases[shape]['weight']
        # Exercise every factorization in unit tests; rings use the one-lane layout.
        residual=torch.randn(1,n,m,device='cuda').transpose(1,2);scale=torch.randn(n,device='cuda')*.01
        def epilogue(y):return residual+y.reshape_as(residual)*scale
        def current(a,b):return epilogue(small(a,b) if family=='fixed' else wide(a,b,bindings))
        methods={'native':lambda a,b:epilogue(F.linear(a,b)), 'current':current,
                 'fused':lambda a,b:fused(a,b,residual,scale,library,bindings)}
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal',x*1e-38,w),('tiny',x*1e-20,w),('large',x*1e20,w),('random',torch.randn_like(x),w),
            ('sparse',x*(torch.arange(k,device='cuda')%29==0),w),
            ('positive_weight_underflow',torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]
        checks=[]
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,b:(fn(a,b),),x,w)
            for label,a,b in variants:
                ref=epilogue(F.linear(a,b));eager=fn(a,b);captured=graph(a,b)[0]
                checks.append({'backend':name,'input':label,'eager':bits(eager,ref),'graph':bits(captured,ref),
                    'layout':eager.stride()==captured.stride()==ref.stride()==residual.stride()})
            del graph
        out,resources=fused(x,w,residual,scale,library,bindings,True)
        record={'shape':shape,'family':family,'config':config,'checks':checks,'resources':resources,'rings':[]}
        assert all(c['eager'] and c['graph'] and c['layout'] for c in checks)
        weights=[w.clone() for _ in range(32)];assert len({v.data_ptr() for v in weights})==32
        for length in (1,8,32):
            funcs={name:lambda z,fn=fn:tuple(fn(z,b) for b in weights[:length]) for name,fn in methods.items() if name!='native'}
            ref=epilogue(F.linear(x,w));ring_checks={};rounds=[]
            for name,fn in funcs.items():
                graph=GraphedCallable(fn,x);ring_checks[name]={'eager':all(bits(y,ref) and y.stride()==ref.stride() for y in fn(x)),
                    'graph':all(bits(y,ref) and y.stride()==ref.stride() for y in graph(x))};del graph
            assert all(c['eager'] and c['graph'] for c in ring_checks.values())
            for repeat in range(3):
                for name in (['current','fused'] if repeat%2==0 else ['fused','current']):
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':do_bench_cudagraph(lambda:funcs[name](x),rep=30)/length})
            medians={name:statistics.median(v['per_call_ms'] for v in rounds if v['backend']==name) for name in funcs}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'checks':ring_checks,'rounds':rounds,'medians_ms':medians,'speedup':medians['current']/medians['fused']})
            print(shape,family,length,record['rings'][-1]['speedup'],flush=True)
        report['records'].append(record);output.write_text(json.dumps(report,indent=2)+'\n')
        del methods,variants,weights,funcs

if __name__=='__main__':main()

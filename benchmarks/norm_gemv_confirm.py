"""Stress and distinct-allocation confirmation for norm/projection fusion."""
import json,statistics,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.norm_gemv import linear,current_norm,SOURCE
from benchmarks.native_layer_norm_tune import bits
from fast_moss.ffn import gemv_linear,math_library
from fast_moss.small_matrices import linear as small
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(401)
    matrix=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];library=math_library()
    tune=json.loads(Path('results/norm_gemv_tune.json').read_text());report={'scope':'exact native Welford/projection fusion; stress and rotating distinct-allocation rings',
        'previous_commit':'b240bd2','revision':REVISION,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for row in tune['records']:
        _,n,k=row['shape'];mode=row['mode'];w=matrix[(1,n,k)]['weight']
        valid=[r for r in row['trials'] if r['bits_equal'] and r['resources']['local_bytes']==0]
        configs=[tuple(r['config']) for r in sorted(valid,key=lambda r:r['ms'])[:3]]
        # Keep the best of each normalization layout represented as a finalist.
        for register in (0,1):configs.append(tuple(min((r for r in valid if r['config'][0]==register),key=lambda r:r['ms'])['config']))
        configs=list(dict.fromkeys(configs))
        def native(a,weight,gg,bb):
            y=F.linear(F.layer_norm(a,(1280,),gg,bb,eps),weight)
            return F.gelu(y) if mode=='gelu' else y
        def current(a,weight,gg,bb):
            z=current_norm(a,gg,bb,eps)
            return gemv_linear(z,weight,'gelu',None,None,library) if mode=='gelu' else small(z,weight)
        methods={'native':native,'current':current}
        methods.update({json.dumps(cfg):lambda a,ww,gg,bb,cfg=cfg:linear(a,ww,gg,bb,eps,mode,cfg) for cfg in configs})
        variants=[('actual',x,w,g,b),('negative',-x,w,g,b),('zero',torch.zeros_like(x),w,g,b),('negative_zero',torch.full_like(x,-0.),w,g,b),
            ('subnormal',x*1e-38,w,g,b),('tiny',x*1e-20,w,g,b),('large',x*1e10,w,g,b),('large_offset',x*.001+10000,w,g,b),
            ('constant',torch.full_like(x,3.5),w,g,b),('random',torch.randn_like(x),w,torch.randn_like(g),torch.randn_like(b)),
            ('zero_affine',x,w,torch.zeros_like(g),torch.zeros_like(b)),('negative_zero_affine',x,w,torch.zeros_like(g),torch.full_like(b,-0.)),
            ('positive_weight_underflow',x,torch.full_like(w,.125),torch.zeros_like(g),torch.full_like(b,-1.401298464324817e-45))]
        record={'shape':row['shape'],'mode':mode,'configs':configs,'checks':[],'rings':[]}
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,ww,gg,bb:(fn(a,ww,gg,bb),),x,w,g,b)
            for label,a,ww,gg,bb in variants:
                ref=native(a,ww,gg,bb)
                c={'backend':name,'input':label,'eager':bits(fn(a,ww,gg,bb),ref),'graph':bits(graph(a,ww,gg,bb)[0],ref)}
                record['checks'].append(c)
                if not c['eager'] or not c['graph']:raise SystemExit(f'Fidelity mismatch {n} {c}')
            del graph
        weights=[w.clone() for _ in range(32)];assert len({ww.data_ptr() for ww in weights})==32
        for length in (1,8,32):
            funcs={name:lambda a,fn=fn:tuple(fn(a,ww,g,b) for ww in weights[:length]) for name,fn in methods.items() if name!='native'}
            ref=native(x,w,g,b);checks={};rounds=[]
            for name,fn in funcs.items():
                graph=GraphedCallable(fn,x);checks[name]={'eager':all(bits(y,ref) for y in fn(x)),'graph':all(bits(y,ref) for y in graph(x))};del graph
            assert all(c['eager'] and c['graph'] for c in checks.values())
            for repeat in range(5):
                names=list(funcs);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:rounds.append({'round':repeat,'backend':name,'per_call_ms':do_bench_cudagraph(lambda:funcs[name](x),rep=30)/length})
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in funcs}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'checks':checks,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/value for name,value in medians.items() if name!='current'}})
            print(n,length,record['rings'][-1]['speedups'],flush=True)
        report['records'].append(record);Path('results/norm_gemv_confirm.json').write_text(json.dumps(report,indent=2)+'\n')
        del weights,funcs,methods,variants

if __name__=='__main__':main()

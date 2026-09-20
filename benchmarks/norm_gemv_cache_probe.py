"""Cache-policy stress and fixed-graph weight-ring comparison."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv_cache import linear as cached,SOURCE
from benchmarks.norm_gemv import linear,current_norm
from benchmarks.native_layer_norm_tune import bits
from fast_moss.ffn import gemv_linear,math_library
from fast_moss.small_matrices import linear as small
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(133)
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];library=math_library()
    selection=json.loads(Path('results/norm_gemv_steady.json').read_text())
    report={'scope':'explicit global cache policies; exact stress and fixed graph 32-allocation rings, 200 warmups, nine samples of ten replays, five rotating rounds',
        'previous_commit':'b240bd2','revision':REVISION,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for row in selection['records']:
        _,n,k=row['shape'];mode=row['mode'];w=matrices[(1,n,k)]['weight'];speeds=row['rings'][-1]['speedups']
        configs=[tuple(json.loads(name)) for name in sorted(speeds,key=speeds.get,reverse=True)[:2]]
        def native(a,ww):
            z=F.linear(F.layer_norm(a,(1280,),g,b,eps),ww)
            return F.gelu(z) if mode=='gelu' else z
        def current(a,ww):
            z=current_norm(a,g,b,eps)
            return gemv_linear(z,ww,'gelu',None,None,library) if mode=='gelu' else small(z,ww)
        methods={'current':current};resources={}
        for cfg in configs:
            methods[json.dumps(cfg)]=lambda a,ww,cfg=cfg:linear(a,ww,g,b,eps,mode,cfg)
            for cache in ['ca','cg','cs']:
                c=(*cfg,cache);name=json.dumps(c)
                methods[name]=lambda a,ww,c=c:cached(a,ww,g,b,eps,mode,c)
                _,resources[name]=cached(x,w,g,b,eps,mode,c,resources=True)
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal',x*1e-38,w),('tiny',x*1e-20,w),('large',x*1e10,w),('large_offset',x*.001+10000,w),('random',torch.randn_like(x),w),('underflow_weight',x,torch.full_like(w,1.401298464324817e-45))]
        checks=[]
        for name,fn in methods.items():
            graph=GraphedCallable(lambda a,ww:(fn(a,ww),),x,w)
            for label,a,ww in variants:
                ref=native(a,ww);c={'backend':name,'input':label,'eager':bits(fn(a,ww),ref),'graph':bits(graph(a,ww)[0],ref)}
                assert c['eager'] and c['graph'];checks.append(c)
            del graph
        weights=[w.clone() for _ in range(32)];rounds=[];ref=native(x,w)
        for repeat in range(5):
            names=list(methods);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                fn=lambda a:tuple(methods[name](a,ww) for ww in weights)
                graph=GraphedCallable(fn,x);assert all(bits(y,ref) for y in graph(x))
                for _ in range(200):graph.graph.replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(9):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                    for _ in range(10):graph.graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/320)
                rounds.append({'round':repeat,'backend':name,'per_call_ms':statistics.median(samples),'samples_ms':samples,'bits_equal':True});del graph
        medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in methods}
        ring={'length':32,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}}
        report['records'].append({'shape':row['shape'],'mode':mode,'configs':configs,'resources':resources,'checks':checks,'rings':[ring]})
        print(n,ring['speedups'],flush=True);Path('results/norm_gemv_cache_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        del weights,methods,variants

if __name__=='__main__':main()

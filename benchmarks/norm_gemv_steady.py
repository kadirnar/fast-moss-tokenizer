"""Fixed-graph replay timing to resolve bimodal unrolled-helper measurements."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv import linear,current_norm,SOURCE
from benchmarks.native_layer_norm_tune import bits
from fast_moss.ffn import gemv_linear,math_library
from fast_moss.small_matrices import linear as small
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision()
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];library=math_library()
    initial=json.loads(Path('results/norm_gemv_confirm.json').read_text())
    report={'scope':'same 32-allocation rings with one fixed graph, 200 warmup replays and nine samples of ten replays; graph copies/output clones excluded',
        'previous_commit':'b240bd2','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for row in initial['records']:
        _,n,k=row['shape'];mode=row['mode'];w=matrices[(1,n,k)]['weight'];weights=[w.clone() for _ in range(32)]
        def current(a,ww):
            z=current_norm(a,g,b,eps)
            return gemv_linear(z,ww,'gelu',None,None,library) if mode=='gelu' else small(z,ww)
        methods={'current':current}
        methods.update({json.dumps(cfg):lambda a,ww,cfg=cfg:linear(a,ww,g,b,eps,mode,tuple(cfg)) for cfg in row['configs']})
        ref=F.linear(F.layer_norm(x,(1280,),g,b,eps),w);ref=F.gelu(ref) if mode=='gelu' else ref
        rounds=[];checks=[]
        for repeat in range(5):
            names=list(methods);names=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in names:
                fn=lambda a:tuple(methods[name](a,ww) for ww in weights)
                graph=GraphedCallable(fn,x)
                exact=all(bits(y,ref) for y in graph(x));assert exact
                for _ in range(200):graph.graph.replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(9):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):graph.graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/320)
                rounds.append({'round':repeat,'backend':name,'per_call_ms':statistics.median(samples),'samples_ms':samples,'bits_equal':exact})
                del graph
        medians={name:statistics.median(t['per_call_ms'] for t in rounds if t['backend']==name) for name in methods}
        ring={'length':32,'logical_weight_bytes':32*w.numel()*4,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}}
        report['records'].append({'shape':row['shape'],'mode':mode,'configs':row['configs'],'rings':[ring]})
        print(n,ring['speedups'],flush=True)
        for name in methods:print(name,[round(r['per_call_ms']*1000,3) for r in rounds if r['backend']==name],flush=True)
        Path('results/norm_gemv_steady.json').write_text(json.dumps(report,indent=2)+'\n')
        del methods,weights

if __name__=='__main__':main()

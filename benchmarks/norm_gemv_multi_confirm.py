"""Exact stress checks and distinct-weight rings for shared staging finalists."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv_multi import linear,SOURCE
from benchmarks.norm_gemv_multi_probe import bits
from fast_moss.norm_async import linear as current,CONFIGS
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(997)
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')]
    probe=json.loads(Path('results/norm_gemv_multi_probe.json').read_text())
    output=Path('results/norm_gemv_multi_confirm.json')
    report={'scope':'research independent output chains; five rotating rounds, fixed graphs, 200 warmups, nine samples of ten replays; copies/clones excluded; distinct allocations are synthetic cache pressure, not physical DRAM measurements',
        'previous_commit':'beb5c32','ranking':'distinct-weight rings, multiple outputs only plus explicit controls','revision':REVISION,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for row in probe['records']:
        _,n,k=row['shape'];mode=row['mode'];w=matrices[(1,n,k)]['weight']
        def ranked(ring):
            return sorted([{'config':tuple(json.loads(name)),'ms':v} for name,v in ring['medians_ms'].items()
                           if name!='current' and json.loads(name)[-1]>1],key=lambda r:r['ms'])
        cold=ranked(row['rings'][1]);warm=ranked(row['rings'][0]);top=cold[0]['config']
        variants=[r['config'] for r in cold[:2]]+[warm[0]['config'],(*CONFIGS[mode],1),(*top[:-1],1),
                  (*top[:6],0,top[7]),(*top[:3],1,*top[4:]),(*top[:7],8)]
        configs=[c for c in dict.fromkeys(variants) if 5120+c[3]*(c[1]//8)*c[7]*(c[2]+c[4])*4+128<=49152]
        methods={'current':lambda a,ww:current(a,ww,g,b,eps,mode)}
        methods.update({json.dumps(c):lambda a,ww,c=c:linear(a,ww,g,b,eps,mode,c) for c in configs})
        resources={};checks=[]
        variants=[('actual',x,w),('negative',-x,w),('zero',torch.zeros_like(x),w),('negative_zero',torch.full_like(x,-0.),w),
            ('subnormal_input',x*1e-38,w),('tiny_input',x*1e-20,w),('large_input',x*1e10,w),('offset',x*.001+10000,w),
            ('random',torch.randn_like(x),w),('subnormal_weight',x,torch.full_like(w,1.401298464324817e-45)),('zero_weight',x,torch.full_like(w,-0.))]
        def native(a,ww):
            z=F.linear(F.layer_norm(a,(1280,),g,b,eps),ww)
            return F.gelu(z) if mode=='gelu' else z
        for name,fn in methods.items():
            if name!='current':_,resources[name]=linear(x,w,g,b,eps,mode,tuple(json.loads(name)),resources=True)
            graph=GraphedCallable(lambda a,ww:(fn(a,ww),),x,w)
            for label,a,ww in variants:
                ref=native(a,ww);c={'backend':name,'input':label,'eager':bits(fn(a,ww),ref),'graph':bits(graph(a,ww)[0],ref)}
                assert c['eager'] and c['graph'],(n,name,label);checks.append(c)
            del graph
        record={'shape':row['shape'],'mode':mode,'configs':configs,'resources':resources,'checks':checks,'rings':[]}
        ref=native(x,w)
        for length in (1,32):
            weights=[w.clone() for _ in range(length)];rounds=[]
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
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/(10*length))
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':statistics.median(samples),'samples_ms':samples,'bits_equal':True});del graph
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in methods}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}})
            print(n,length,record['rings'][-1]['speedups'],flush=True);del weights
        report['records'].append(record);output.write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=True;output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

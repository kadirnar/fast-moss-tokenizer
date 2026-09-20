"""Stress and fixed-ring confirmation of residual GEMV staging finalists."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.residual_gemv_async import linear,SOURCE,shared_bytes
from benchmarks.residual_gemv_async_probe import bits,current
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(1007);matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    ring=json.loads(Path('results/residual_gemv_async_ring.json').read_text());probe=json.loads(Path('results/residual_gemv_async_probe.json').read_text())
    report={'scope':'research residual GEMV staging; five rotating rounds, 200 warmups, nine samples of ten fixed graph replays; copies/clones excluded; distinct allocations model cache pressure, not physical DRAM traffic',
        'previous_commit':'b2a1086','revision':REVISION,'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for row in ring['records']:
        k=row['shape'][2];x,w=[matrices[(1,1280,k)][s] for s in ('x','weight')]
        r=torch.randn(1,1280,device='cuda');s=torch.randn(1280,device='cuda')*.1
        ranked=sorted(row['speedups'],key=row['speedups'].get,reverse=True);top=tuple(json.loads(ranked[0]))
        warm=tuple(min(next(t['trials'] for t in probe['records'] if t['shape']==row['shape']),key=lambda r:r['ms'])['config'])
        variants=[tuple(json.loads(v)) for v in ranked[:2]]+[warm,(*top[:5],0,top[6]),(*top[:2],1,*top[3:]),(*top[:4],16,*top[5:]),(*top[:3],16-top[3],*top[4:])]+[(*top[:6],v) for v in (0,1,2)]
        configs=[c for c in dict.fromkeys(variants) if shared_bytes(k,c)<=49152]
        methods={'current':current};methods.update({json.dumps(c):lambda a,ww,rr,ss,c=c:linear(a,ww,rr,ss,c) for c in configs})
        projection=F.linear(x,w)
        stress=[('actual',x,w,r,s),('negative',-x,w,r,s),('zero',torch.zeros_like(x),w,r,s),
            ('negative_zero',torch.full_like(x,-0.),w,torch.full_like(r,-0.),torch.ones_like(s)),
            ('subnormal_input',x*1e-38,w,torch.zeros_like(r),torch.ones_like(s)),
            ('tiny_input',x*1e-20,w,torch.zeros_like(r),torch.ones_like(s)),('large_input',x*1e10,w,r,s),
            ('random',torch.randn_like(x),w,r,s),('positive_underflow',x.abs()*1e-38,w.abs(),torch.zeros_like(r),torch.ones_like(s)),
            ('zero_scale',x,w,torch.full_like(r,-0.),torch.full_like(s,-0.)),
            ('subnormal_scale',x,w,torch.zeros_like(r),torch.full_like(s,1.401298464324817e-45)),
            ('subnormal_weight',torch.ones_like(x),torch.full_like(w,1.401298464324817e-45),torch.zeros_like(r),torch.ones_like(s)),
            ('cancel',x,w,-projection,torch.ones_like(s)),('scale_cancel',x,w,-projection*s,s)]
        checks=[];resources={}
        for name,fn in methods.items():
            if name!='current':_,resources[name]=linear(x,w,r,s,tuple(json.loads(name)),resources=True)
            graph=GraphedCallable(lambda a,ww,rr,ss:(fn(a,ww,rr,ss),),x,w,r,s)
            for label,a,ww,rr,ss in stress:
                ref=rr+F.linear(a,ww)*ss;c={'backend':name,'input':label,'eager':bits(fn(a,ww,rr,ss),ref),'graph':bits(graph(a,ww,rr,ss)[0],ref)}
                assert c['eager'] and c['graph'],(k,name,label);checks.append(c)
            del graph
        record={'shape':row['shape'],'configs':configs,'resources':resources,'checks':checks,'rings':[]};ref=current(x,w,r,s)
        for length in (1,32):
            weights=[w.clone() for _ in range(length)];rounds=[]
            for repeat in range(5):
                names=list(methods);names=names[repeat%len(names):]+names[:repeat%len(names)]
                for name in names:
                    graph=GraphedCallable(lambda a:tuple(methods[name](a,ww,r,s) for ww in weights),x);assert all(bits(y,ref) for y in graph(x))
                    for _ in range(200):graph.graph.replay()
                    torch.cuda.synchronize();samples=[]
                    for _ in range(9):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                        for _ in range(10):graph.graph.replay()
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/(10*length))
                    rounds.append({'round':repeat,'backend':name,'per_call_ms':statistics.median(samples),'samples_ms':samples,'bits_equal':True});del graph
            medians={name:statistics.median(t['per_call_ms'] for t in rounds if t['backend']==name) for name in methods}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}})
            print(k,length,record['rings'][-1]['speedups'],flush=True);del weights
        report['records'].append(record);Path('results/residual_gemv_async_confirm.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=True;Path('results/residual_gemv_async_confirm.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

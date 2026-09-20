"""Rank every exact staging candidate under a 32-allocation weight ring."""
import json,statistics
from pathlib import Path
import torch
from benchmarks.norm_gemv_async import linear
from benchmarks.norm_gemv_async_probe import bits
from fast_moss.norm_projection import linear as current
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision

@torch.inference_mode()
def main():
    strict_precision();matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')]
    probe=json.loads(Path('results/norm_gemv_async_probe.json').read_text())
    report={'scope':'all 152 exact schedules; 32 distinct weight allocations; two reversed-order rounds, 50 warmups, three samples of five replays, copies/clones excluded; exploratory selection only','previous_commit':'b40b3b4','records':[]}
    for row in probe['records']:
        _,n,k=row['shape'];mode=row['mode'];w=matrices[(1,n,k)]['weight'];weights=[w.clone() for _ in range(32)]
        methods={'current':lambda a,ww:current(a,ww,g,b,eps,mode)}
        methods.update({json.dumps(c['config']):lambda a,ww,c=tuple(c['config']):linear(a,ww,g,b,eps,mode,c) for c in row['trials']})
        ref=current(x,w,g,b,eps,mode);rounds=[]
        for repeat in range(2):
            names=list(methods);names=names if repeat==0 else list(reversed(names))
            for name in names:
                graph=GraphedCallable(lambda a:tuple(methods[name](a,ww) for ww in weights),x)
                assert all(bits(z,ref) for z in graph(x))
                for _ in range(50):graph.graph.replay()
                torch.cuda.synchronize();samples=[]
                for _ in range(3):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                    for _ in range(5):graph.graph.replay()
                    end.record();end.synchronize();samples.append(start.elapsed_time(end)/160)
                rounds.append({'round':repeat,'backend':name,'samples_ms':samples,'per_call_ms':statistics.median(samples),'bits_equal':True});del graph
        medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in methods}
        record={'shape':row['shape'],'mode':mode,'length':32,'logical_weight_bytes':32*w.numel()*4,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}}
        report['records'].append(record);Path('results/norm_gemv_async_ring.json').write_text(json.dumps(report,indent=2)+'\n')
        print(n,'current',medians['current'],'top',sorted(record['speedups'].items(),key=lambda r:r[1],reverse=True)[:5],flush=True);del weights
    report['all_exact']=True;Path('results/norm_gemv_async_ring.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

"""Explore independent output chains with captured exactness and weight rings."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv_multi import linear,SOURCE
from fast_moss.norm_async import linear as current,CONFIGS
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION


def bits(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))

def configs():
    return [(reg,threads,tile,2,pad,16,1,outputs)
        for reg,threads in [(1,32),(1,64),(0,128),(1,128),(1,256)]
        for tile in (32,64,128,256,320) for pad in (0,8) for outputs in (1,2,4,8)
        if 5120+2*(threads//8)*outputs*(tile+pad)*4+128<=49152]

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(1014)
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];normalized=F.layer_norm(x,(1280,),g,b,eps)
    report={'scope':'exploratory independent output chains per thread group; captured normalized/output bits; two reverse-order rounds, 50 warmups and three samples of five fixed graph replays per ring; copies/clones excluded; distinct allocations model cache pressure, not physical DRAM traffic',
        'previous_commit':'beb5c32','revision':REVISION,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for n,mode in [(3840,'none'),(5120,'gelu')]:
        w=matrices[(1,n,1280)]['weight'];ref=F.linear(normalized,w);ref=F.gelu(ref) if mode=='gelu' else ref
        assert bits(current(x,w,g,b,eps,mode),ref)
        trials=[]
        for cfg in configs():
            y,z=linear(x,w,g,b,eps,mode,cfg,debug=True);out,res=linear(x,w,g,b,eps,mode,cfg,resources=True)
            assert bits(y,ref) and bits(z,normalized) and bits(out,ref),(mode,cfg)
            trials.append({'config':cfg,'bits_equal':True,'resources':res,'grid_blocks':n//((cfg[1]//8)*cfg[-1])})
        methods={'current':lambda a,ww:current(a,ww,g,b,eps,mode)}
        methods.update({json.dumps(t['config']):lambda a,ww,c=tuple(t['config']):linear(a,ww,g,b,eps,mode,c) for t in trials})
        record={'shape':[1,n,1280],'mode':mode,'current_config':CONFIGS[mode],'trials':trials,'rings':[]}
        for length in (1,32):
            weights=[w.clone() for _ in range(length)];rounds=[]
            for repeat in range(2):
                names=list(methods);names=names if repeat==0 else list(reversed(names))
                for name in names:
                    graph=GraphedCallable(lambda a:tuple(methods[name](a,ww) for ww in weights),x)
                    assert all(bits(y,ref) for y in graph(x))
                    for _ in range(50):graph.graph.replay()
                    torch.cuda.synchronize();samples=[]
                    for _ in range(3):
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                        for _ in range(5):graph.graph.replay()
                        end.record();end.synchronize();samples.append(start.elapsed_time(end)/(5*length))
                    rounds.append({'round':repeat,'backend':name,'samples_ms':samples,'per_call_ms':statistics.median(samples),'bits_equal':True});del graph
            medians={name:statistics.median(r['per_call_ms'] for r in rounds if r['backend']==name) for name in methods}
            record['rings'].append({'length':length,'logical_weight_bytes':length*w.numel()*4,'rounds':rounds,'medians_ms':medians,'speedups':{name:medians['current']/v for name,v in medians.items() if name!='current'}})
            print(mode,'schedules',len(trials),'ring',length,'current',medians['current'],'top',sorted(record['rings'][-1]['speedups'].items(),key=lambda r:r[1],reverse=True)[:5],flush=True)
            del weights
        report['records'].append(record);Path('results/norm_gemv_multi_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=True;Path('results/norm_gemv_multi_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

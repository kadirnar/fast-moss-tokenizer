"""Arithmetic/layout/compiler and warmed graph timing gates for attention epilogues."""
import hashlib,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.attention_residual import CONFIGS,SOURCE,linear,current
from benchmarks.native_layer_norm_tune import bits
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(953)
    captured=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research native attention output projection/residual fusion, captured matrices and synthetic residuals/scales','previous_commit':'84caafc','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'cuda_source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'timing_scope':'32-allocation fixed graph rings, 200 warmups, five rotating rounds of nine ten-replay samples; copies/clones excluded; synthetic cache pressure, not DRAM counters','records':[]}
    for shape in sorted(CONFIGS):
        m,n,k=shape;x=captured[shape]['x'];w=captured[shape]['weight'];scale=torch.randn(n,device='cuda')*.1
        for time in [1]+[v for v in range(2,m+1) if m%v==0]:
            residual=torch.randn(m//time,n,time,device='cuda').transpose(1,2) if time>1 else torch.randn(m,1,n,device='cuda')
            fn=lambda z,r,s:linear(z,w,r,s)
            graph=GraphedCallable(lambda z,r,s:(fn(z,r,s),),x,residual,scale);checks=0
            for a,r,s in [(x,residual,scale),(x*1e-38,residual*0,scale),(x*1e10,residual,scale),
                          (torch.full_like(x,-0.),torch.full_like(residual,-0.),scale),
                          (x,residual,torch.zeros_like(scale)),(x,residual,torch.full_like(scale,1e-38))]:
                ref=r+F.linear(a,w).reshape_as(r)*s
                for out in [fn(a,r,s),graph(a,r,s)[0],current(a,w,r,s)]:
                    assert bits(out,ref),(shape,time,'bits');assert out.stride()==ref.stride(),(shape,time,'stride');checks+=1
            out,resources=linear(x,w,residual,scale,True)
            assert resources['local_bytes']==resources['local_loads']==resources['local_stores']==resources['matrix_instructions']==0
            del graph
            record={'shape':shape,'time':time,'config':CONFIGS[shape],'comparisons':checks,'all_bits_equal':True,'output_stride':list(out.stride()),'resources':resources}
            # Full layout arithmetic coverage; time the two endpoints only.
            if time in (1,m):
                weights=[w.clone() for _ in range(32)];rounds=[]
                for repeat in range(5):
                    methods=[('current',current),('fused',linear)]
                    if repeat%2:methods.reverse()
                    for name,helper in methods:
                        graph=GraphedCallable(lambda z:tuple(helper(z,ww,residual,scale) for ww in weights),x)
                        ref=residual+F.linear(x,w).reshape_as(residual)*scale
                        assert all(bits(y,ref) for y in graph(x))
                        for _ in range(200):graph.graph.replay()
                        torch.cuda.synchronize();samples=[]
                        for _ in range(9):
                            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                            for _ in range(10):graph.graph.replay()
                            end.record();end.synchronize();samples.append(start.elapsed_time(end)/320)
                        rounds.append({'round':repeat,'backend':name,'samples_ms':samples});del graph
                medians={name:statistics.median(v for row in rounds if row['backend']==name for v in row['samples_ms']) for name in ('current','fused')}
                record.update(rounds=rounds,medians_ms=medians,speedup=medians['current']/medians['fused'],logical_weight_bytes=32*w.numel()*4)
                del weights
            report['records'].append(record);Path('results/attention_residual_probe.json').write_text(json.dumps(report,indent=2)+'\n')
            print(shape,time,'exact',record.get('speedup'),flush=True)
    report['all_exact']=True;report['comparisons']=sum(r['comparisons'] for r in report['records']);Path('results/attention_residual_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

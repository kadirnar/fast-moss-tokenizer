"""Every learned LFQ output weight: native convolution, fused FMA, and table gates."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.compare import difference
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.projections import project
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    model=load_model()
    report={'scope':'all 32 actual LFQ output projection weights; synthetic activation stress and all learned codebook entries',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'cudnn':torch.backends.cudnn.version(),'dtype':'float32','tf32':False,'records':[],'tables':[]}
    with optimized(model):
        for i,q in enumerate(model.quantizer.quantizers):
            w,bias=q.out_proj._fast_weight,q.out_proj.bias
            canonical=q.codebook.weight[:,:,None].contiguous()
            conv=F.conv1d(canonical,w,bias)[:,:,0]
            cache=F.conv1d(q.codebook.weight.T[None],w,bias)[0].T
            report['tables'].append({'quantizer':i,'native_different_geometry':difference(conv,cache),
                                      'fused_all_entries':difference(conv,project(canonical,w,bias)[:,:,0])})
            for b,t in [(1,1),(1,3),(8,3),(128,3),(8,40)]:
                torch.manual_seed(456+i)
                source=torch.randn(b,8,t,device='cuda')
                checks={}
                for name,scale in [('zero',0.),('normal',.2),('tiny',1e-38),('large',1e12)]:
                    x=source*scale
                    checks[name]=difference(F.conv1d(x,w,bias),project(x,w,bias))
                x=source*.2
                graph=GraphedCallable(lambda z:(project(z,w,bias),),x)
                checks['graph']=difference(F.conv1d(x,w,bias),graph(x)[0])
                del graph
                report['records'].append({'quantizer':i,'shape':[b,8,t],'comparisons':checks})
            print('quantizer',i,'exact',all(v['exact'] for r in report['records'][-5:] for v in r['comparisons'].values()),flush=True)
    report['all_exact']=(all(v['exact'] for r in report['records'] for v in r['comparisons'].values())
                          and all(v['exact'] for r in report['tables'] for k,v in r.items() if k!='quantizer'))
    Path('results/projection_components.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Projection component fidelity failed')


if __name__=='__main__':main()

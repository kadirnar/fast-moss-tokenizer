"""Stress and paired confirmation for pipeline depth and fused FFN epilogues."""
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
import nvidia.cuda_nvcc
from triton.testing import do_bench_cudagraph
from fast_moss.ordered_matrices import _partials,_reduce,linear as baseline
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from benchmarks.ordered_epilogue import linear as fused
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay


def staged(x,packed):
    m,k=x.shape;n=packed.shape[1]
    partials=torch.empty((k//256,m,n),device=x.device);out=torch.empty((m,n),device=x.device)
    _partials[(1,n//128,k//256)](x,packed,partials,m,n,k,256,32,128,32,num_warps=4,
                                   num_stages=2,enable_fp_fusion=False)
    _reduce[(m*n//256,)](partials,out,m*n,k//256,256,enable_fp_fusion=False)
    return out


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(745)
    library=str(Path(nvidia.cuda_nvcc.__path__[0])/'nvvm/libdevice/libdevice.10.bc')
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'FP32 FFN component stress and three alternating warm/cold paired rounds',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':745,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for shape,mode in [((24,5120,1280),'gelu'),((24,1280,5120),'residual')]:
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        residual=torch.randn(24,shape[1],device='cuda');scale=torch.randn(shape[1],device='cuda')
        post=(lambda z:F.gelu(z)) if mode=='gelu' else (lambda z:residual+z*scale)
        fns={'previous':lambda z:post(baseline(z,packed)),
             'stages2':lambda z:post(staged(z,packed)),
             'fused_stages3':lambda z:fused(z,packed,mode,residual,scale,stages=3,math_library=library),
             'fused_stages2':lambda z:fused(z,packed,mode,residual,scale,stages=2,math_library=library)}
        variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
                  ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
                  ('large',torch.randn_like(x)*1e20)]
        variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
        sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
        checks=[]
        for name,fn in fns.items():
            graph=GraphedCallable(lambda z:(fn(z),),x)
            for label,z in variants:
                ref=post(F.linear(z,w));out=fn(z);replay=graph(z)[0]
                checks.append({'backend':name,'input':label,'eager':difference(ref,out),'graph':difference(ref,replay),
                               'eager_bits_equal':torch.equal(ref.view(torch.int32),out.view(torch.int32))})
            del graph
        rounds=[]
        for repeat in range(3):
            names=list(fns)
            if repeat%2:names.reverse()
            for name in names:
                fn=fns[name];graph=GraphedCallable(lambda z:(fn(z),),x)
                rounds.append({'round':repeat,'backend':name,'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=30),
                               'cold':evicted_replay(graph,flush,repeats=25)})
                del graph
        medians={name:{'warm':statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                       'cold':statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)} for name in fns}
        r={'shape':shape,'mode':mode,'checks':checks,'timings':rounds,'medians_ms':medians,
           'all_exact':all(c[k]['exact'] for c in checks for k in ['eager','graph'])}
        report['records'].append(r);print(shape,r['all_exact'],medians,flush=True)
        Path('results/ffn_components.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(r['all_exact'] for r in report['records'])
    Path('results/ffn_components.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('FFN component gate failed')


if __name__=='__main__':main()

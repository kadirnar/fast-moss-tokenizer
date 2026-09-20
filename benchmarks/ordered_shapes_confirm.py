"""Stress and repeat the expanded ordered-matrix candidates using runtime kernels."""
import argparse
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench_cudagraph
from fast_moss.ordered_matrices import _partials, _reduce
from fast_moss.cublaslt import LinearPlan
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay


def linear(x,packed,config,return_kernel=False):
    m,k=x.shape;n=packed.shape[1]
    chunk,bm,bn,bk,warps,stages=config
    parts=triton.cdiv(k,chunk)
    p=torch.empty((parts,m,n),device=x.device,dtype=x.dtype)
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    kernel=_partials[(triton.cdiv(m,bm),triton.cdiv(n,bn),parts)](
        x,packed,p,m,n,k,chunk,bm,bn,bk,num_warps=warps,
        num_stages=stages,enable_fp_fusion=False)
    _reduce[(triton.cdiv(m*n,256),)](p,out,m*n,parts,256,enable_fp_fusion=False)
    return (out,kernel) if return_kernel else out


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--search',default='results/ordered_shapes.json')
    parser.add_argument('--output',default='results/ordered_shapes_confirm.json')
    args=parser.parse_args()
    strict_precision();torch.manual_seed(6318)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    search=json.loads(Path(args.search).read_text())
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text())
    selected={tuple(r['shape']):r for r in profile['records']}
    report={'scope':'expanded ordered FP32 actual-weight stress, three alternating warm/cold rounds',
            'revision':REVISION,'torch':torch.__version__,'triton':triton.__version__,
            'gpu':torch.cuda.get_device_name(),'seed':6318,'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    for record in search['records']:
        tiles=sorted([t for t in record['tiles'] if t.get('difference',{}).get('exact')],
                     key=lambda t:t['warm_ms'])[:2]
        if not tiles:continue
        shape=tuple(record['shape']);x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        configs={str((t['chunk'],*t['tile'],stage)):(t['chunk'],*t['tile'],stage)
                 for t in tiles for stage in [2,3]}
        with LinearPlan(w,shape[0],'packed',packed_weight=packed) as plan:
            if shape in selected:
                index=plan.restore(selected[shape]['algorithm'],profile['cublaslt_version'])
                previous=lambda z:plan(z,index)
            else:previous=lambda z:F.linear(z,w)
            fns={'previous':previous,**{name:(lambda z,cfg=cfg:linear(z,packed,cfg)) for name,cfg in configs.items()}}
            variants=[('actual',x),('negated',-x),('scaled',x*.17),('zero',torch.zeros_like(x)),
                      ('subnormal',torch.randn_like(x)*1e-38),('tiny',torch.randn_like(x)*1e-20),
                      ('large',torch.randn_like(x)*1e20)]
            variants += [(f'random_{i}',torch.randn_like(x)) for i in range(4)]
            sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127]);variants.append(('sparse',sparse))
            checks=[];resources={}
            for name,fn in fns.items():
                if name in configs:
                    _,kernel=linear(x,packed,configs[name],True)
                    resources[name]={'registers':kernel.n_regs,'spills':kernel.n_spills,
                                     'shared_bytes':kernel.metadata.shared}
                graph=GraphedCallable(lambda z:(fn(z),),x)
                for label,z in variants:
                    ref=F.linear(z,w);out=fn(z);replay=graph(z)[0]
                    checks.append({'backend':name,'input':label,'eager':difference(ref,out),
                        'graph':difference(ref,replay),'eager_bits_equal':torch.equal(ref.view(torch.int32),out.view(torch.int32))})
                del graph
            exact={name:all(c[k]['exact'] and c['eager_bits_equal'] for c in checks if c['backend']==name
                            for k in ['eager','graph']) for name in fns}
            rounds=[]
            for repeat in range(3):
                names=list(fns)
                if repeat%2:names.reverse()
                for name in names:
                    if not exact[name]:continue
                    fn=fns[name];graph=GraphedCallable(lambda z:(fn(z),),x)
                    rounds.append({'round':repeat,'backend':name,'warm_ms':do_bench_cudagraph(lambda:fn(x),rep=30),
                                   'cold':evicted_replay(graph,flush,repeats=25)})
                    del graph
            medians={name:{'warm':statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                           'cold':statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)}
                     for name in fns if exact[name]}
            r={'shape':shape,'configs':configs,'checks':checks,'exact':exact,'resources':resources,
               'timings':rounds,'medians_ms':medians,'all_exact':all(exact.values())}
            report['records'].append(r)
            print(shape,'exact',exact,'medians',medians,flush=True)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(r['all_exact'] for r in report['records'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

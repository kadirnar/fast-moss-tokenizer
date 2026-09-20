"""Stress selected actual-weight FP32 algorithms beyond their tuning activation."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.cublaslt_model import choices
from benchmarks.compare import difference
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',default='results/matrix_inputs.pt')
    p.add_argument('--tuning',nargs='+',default=['results/matrices_cublaslt.json','results/matrices_cublaslt_codec8.json'])
    p.add_argument('--output',default='results/cublaslt_fidelity.json')
    a=p.parse_args();strict_precision();torch.manual_seed(3812)
    cases=torch.load(a.inputs,weights_only=True);selected=choices(a.tuning)
    report={'scope':'actual-weight stress checks, not a universal exactness proof',
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cublaslt_version':library().cublasLtGetVersion(),'seed':3812,'records':[]}
    for shape,record in selected.items():
        case=cases[shape];w=case['weight'];x=case['x']
        with LinearPlan(w,shape[0],record['backend'].split('_')[1]) as plan:
            index=plan.restore(record['algorithm'],record['cublaslt_version'])
            variants=[('actual',x),('negated',-x),('scaled_actual',x*.17),
                      ('zeros',torch.zeros_like(x)),('subnormal',torch.randn_like(x)*1e-38),
                      ('tiny',torch.randn_like(x)*1e-20),('large',torch.randn_like(x)*1e20)]
            variants.extend((f'random_{i}',torch.randn_like(x)) for i in range(4))
            sparse=torch.zeros_like(x);sparse[:,::127]=torch.randn_like(sparse[:,::127])
            variants.append(('sparse',sparse))
            graph=GraphedCallable(lambda z:(plan(z,index),),x)
            checks=[]
            for name,z in variants:
                ref=F.linear(z,w);out=plan(z,index)
                checks.append({'input':name,'reference':difference(ref,out),
                               'graph_vs_eager':difference(out,graph(z)[0])})
            del graph
        r={'shape_MNK':shape,'backend':record['backend'],'checks':checks}
        report['records'].append(r)
        print(shape,'exact',all(c['reference']['exact'] and c['graph_vs_eager']['exact'] for c in checks),flush=True)
    report['all_exact']=all(c['reference']['exact'] and c['graph_vs_eager']['exact'] for r in report['records'] for c in r['checks'])
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

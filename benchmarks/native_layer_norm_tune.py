"""Actual-operand LayerNorm layout sweep with native-output and statistic gates."""
import argparse
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from benchmarks.native_layer_norm import layer_norm
from fast_moss.loading import strict_precision,REVISION


def bits(a,b):return torch.equal(a.reshape(-1).view(torch.int32),b.reshape(-1).view(torch.int32))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fixed',action='store_true')
    parser.add_argument('--fixed-mode',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--output',default='results/native_layer_norm_tune.json');args=parser.parse_args()
    strict_precision();cases=torch.load('results/layer_norm_inputs.pt',weights_only=True)
    report={'scope':'actual LayerNorm operands, native arithmetic CUDA layout search',
        'revision':REVISION,'previous_commit':'52a2c91','torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape,row in sorted(cases.items()):
        x,g,b,eps=(row[k] for k in ('x','weight','bias','eps'))
        ref=torch.native_layer_norm(x,(shape[1],),g,b,eps)
        assert bits(ref[0],row['output'])
        record={'shape':shape,'native_ms':do_bench_cudagraph(lambda:torch.native_layer_norm(x,(shape[1],),g,b,eps),rep=30),'trials':[]}
        configs=[(False,128,u,True) for u in (1,4)]+[(True,t,u,True) for t in (32,64,128,256) for u in (1,4)]
        if args.fixed:configs=[(*c,args.fixed_mode) for c in configs]
        for cfg in configs:
            out=layer_norm(x,g,b,eps,cfg,True)
            checks=[bits(a,v) for a,v in zip(ref,out)]
            record['trials'].append({'config':cfg,'bits_equal':checks,'resources':out[-1],
                'ms':do_bench_cudagraph(lambda:layer_norm(x,g,b,eps,cfg),rep=20)})
        best=min(record['trials'],key=lambda t:t['ms'])
        print(shape,'native',record['native_ms'],'best',best['config'],best['ms'],
              'ratio',record['native_ms']/best['ms'],'exact',all(all(t['bits_equal']) for t in record['trials']),flush=True)
        report['records'].append(record)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

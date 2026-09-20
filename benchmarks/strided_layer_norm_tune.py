"""Captured-layout CUDA LayerNorm sweep: native copy+norm versus one kernel."""
import json
from pathlib import Path
import torch
from triton.testing import do_bench_cudagraph
from benchmarks.strided_layer_norm import layer_norm,compile_kernel
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/strided_layer_norm_inputs.pt',weights_only=True)
    report={'scope':'actual noncontiguous LayerNorm operands, native copy+norm included',
        'revision':REVISION,'previous_commit':'6bdd709','torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'records':[]}
    configs=[(False,128,u,False,0) for u in (1,4)]+[(True,t,u,False,0) for t in (32,64,128,256) for u in (1,4)]
    configs += [(True,t,u,True,p) for t in (64,128,256) for u in (1,4) for p in (0,4)]
    for key,row in sorted(cases.items()):
        x,g,b,eps=(row[k] for k in ('x','weight','bias','eps'));n=x.shape[-1]
        native=lambda:torch.native_layer_norm(x,(n,),g,b,eps)
        ref=native();assert bits(ref[0],row['output'])
        record={'shape':list(x.shape),'stride':list(x.stride()),'native_ms':do_bench_cudagraph(native,rep=20),'trials':[]}
        for cfg in configs:
            out=layer_norm(x,g,b,eps,cfg);checks=[bits(a,v) for a,v in zip(ref,out)]
            record['trials'].append({'config':cfg,'bits_equal':checks,'resources':compile_kernel(n,x.shape[1],cfg)[2],
                'ms':do_bench_cudagraph(lambda:layer_norm(x,g,b,eps,cfg),rep=15)})
        valid=[v for v in record['trials'] if all(v['bits_equal'])]
        best=min(valid,key=lambda v:v['ms']) if valid else None
        print(key,'exact',len(valid),'/',len(configs),'best',best,'native',record['native_ms'],flush=True)
        report['records'].append(record);Path('results/strided_layer_norm_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

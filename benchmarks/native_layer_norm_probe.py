"""Arithmetic gate before promoting a new CUDA normalization layout."""
import json
from pathlib import Path
import torch
from benchmarks.native_layer_norm import layer_norm
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable
from benchmarks.compare import difference
from triton.testing import do_bench_cudagraph


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(8439)
    report={'scope':'synthetic native LayerNorm arithmetic/layout probe','previous_commit':'52a2c91','records':[]}
    for n in (768,1280):
        x=torch.randn(8,n,device='cuda');g=torch.randn(n,device='cuda');b=torch.randn(n,device='cuda')
        ref=torch.native_layer_norm(x,(n,),g,b,1e-5)
        for register,threads,u,fmad in [(False,128,1,True),(False,128,1,False),(True,128,1,True),(True,128,1,False),(True,128,4,True)]:
            cfg=(register,threads,u,fmad);out=layer_norm(x,g,b,config=cfg,resources=True)
            checks=[difference(a.reshape(-1),v.reshape(-1)) for a,v in zip(ref,out)]
            bits=[torch.equal(a.reshape(-1).view(torch.int32),v.reshape(-1).view(torch.int32)) for a,v in zip(ref,out)]
            record={'n':n,'config':cfg,'checks':checks,'bits':bits,'resources':out[-1]}
            report['records'].append(record);print(n,cfg,bits,[(c['max_abs'],c['different_elements']) for c in checks],flush=True)
            Path('results/native_layer_norm_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

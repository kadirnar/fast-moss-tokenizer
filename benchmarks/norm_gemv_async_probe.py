"""Explore asynchronous staging without changing production dispatch."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.norm_gemv_async import linear,SOURCE
from fast_moss.norm_projection import linear as current
from fast_moss.loading import strict_precision,REVISION

def bits(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(995)
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];normalized=F.layer_norm(x,(1280,),g,b,eps)
    configs=[(reg,threads,tile,stages,pad,4,1)
        for reg,threads in [(1,32),(1,64),(0,128),(1,128),(1,256)]
        for tile in (64,128,256,320) for stages in (1,2) for pad in (0,8)
        if 5120+stages*(threads//8)*(tile+pad)*4+128<=49152]
    report={'scope':'exploratory exact norm/GEMV shared weight staging; warm graph timings only','previous_commit':'b40b3b4','revision':REVISION,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for n,mode in [(3840,'none'),(5120,'gelu')]:
        w=matrices[(1,n,1280)]['weight'];ref=F.linear(normalized,w);ref=F.gelu(ref) if mode=='gelu' else ref
        assert bits(current(x,w,g,b,eps,mode),ref)
        row={'shape':[1,n,1280],'mode':mode,'current_ms':do_bench_cudagraph(lambda:current(x,w,g,b,eps,mode),rep=40),'trials':[]}
        for cfg in configs:
            y,z=linear(x,w,g,b,eps,mode,cfg,debug=True)
            out,res=linear(x,w,g,b,eps,mode,cfg,resources=True)
            exact=bits(y,ref) and bits(out,ref) and bits(z,normalized)
            assert exact,(n,cfg)
            row['trials'].append({'config':cfg,'bits_equal':exact,'resources':res,'ms':do_bench_cudagraph(lambda:linear(x,w,g,b,eps,mode,cfg),rep=20)})
        report['records'].append(row);Path('results/norm_gemv_async_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        print(n,'current',row['current_ms'],'best',sorted(row['trials'],key=lambda r:r['ms'])[:3],flush=True)
    report['all_exact']=True;Path('results/norm_gemv_async_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

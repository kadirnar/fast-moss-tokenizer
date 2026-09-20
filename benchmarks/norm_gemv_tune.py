"""Explore shared normalized-activation staging inside exact one-row projections."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.norm_gemv import linear,current_norm,SOURCE
from benchmarks.native_layer_norm_tune import bits
from fast_moss.small_matrices import linear as small
from fast_moss.ffn import gemv_linear,math_library
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(728)
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')];normalized=F.layer_norm(x,(1280,),g,b,eps)
    assert bits(normalized,current_norm(x,g,b,eps));library=math_library()
    report={'scope':'research fused native Welford and cyclic GEMV; captured norm1 operand with independently captured projection weights; exploratory warm graph timings',
        'previous_commit':'b240bd2','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'norm_source':norm['name'],'records':[]}
    for n,mode in [(3840,'none'),(5120,'gelu')]:
        w=matrices[(1,n,1280)]['weight'];ref=F.linear(normalized,w);ref=F.gelu(ref) if mode=='gelu' else ref
        def current():
            z=current_norm(x,g,b,eps)
            return gemv_linear(z,w,'gelu',None,None,library) if mode=='gelu' else small(z,w)
        assert bits(current(),ref)
        record={'shape':[1,n,1280],'mode':mode,'current_ms':do_bench_cudagraph(current,rep=50),'trials':[]}
        configs=[(register,vec,threads,p,u) for register,threads in [(0,128),(1,32),(1,64),(1,128),(1,256)] for vec in (1,2,4,8) for p in (1,2,4) for u in (4,16)]
        for cfg in configs:
            (y,z),_=linear(x,w,g,b,eps,mode,cfg,True,True)
            y,res=linear(x,w,g,b,eps,mode,cfg,resources=True)
            exact=bits(y,ref) and bits(z,normalized)
            record['trials'].append({'config':cfg,'bits_equal':exact,'resources':res,'ms':do_bench_cudagraph(lambda:linear(x,w,g,b,eps,mode,cfg),rep=20)})
            if not exact:raise SystemExit(f'Fidelity mismatch {n} {cfg}')
        report['records'].append(record);Path('results/norm_gemv_tune.json').write_text(json.dumps(report,indent=2)+'\n')
        best=min(record['trials'],key=lambda r:r['ms']);print(n,mode,'current',record['current_ms'],'best',best,flush=True)

if __name__=='__main__':main()

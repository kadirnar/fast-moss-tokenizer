"""Exact captured-operand sweep for shared residual GEMV scheduling."""
import hashlib,json
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.residual_gemv_async import linear,SOURCE,shared_bytes
from fast_moss.ffn import gemv_linear,math_library
from fast_moss.attention_residual import linear as attention_linear
from fast_moss.loading import strict_precision,REVISION

_math_library=lru_cache(maxsize=1)(math_library)

def bits(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))
def current(x,w,residual,scale):
    return (gemv_linear(x,w,'residual',residual,scale,_math_library()) if w.shape[1]==5120
            else attention_linear(x,w,residual.reshape(1,1,1280),scale).reshape(1,1280))

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(1005);matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'exploratory native residual GEMV, exact captured matrix plus synthetic residual/scale, warm graph timing only',
        'previous_commit':'b2a1086','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for k in (1280,5120):
        x,w=[matrices[(1,1280,k)][s] for s in ('x','weight')];r=torch.randn(1,1280,device='cuda');s=torch.randn(1280,device='cuda')*.1
        ref=r+F.linear(x,w)*s;assert bits(current(x,w,r,s),ref)
        configs=[(threads,tile,2,pad,4,1,inputs) for threads in (32,64,128,256) for tile in (64,128,256,320)
            for pad in (0,16) for inputs in (0,1,2) if shared_bytes(k,(threads,tile,2,pad,4,1,inputs))<=49152]
        row={'shape':[1,1280,k],'current_ms':do_bench_cudagraph(lambda:current(x,w,r,s),rep=40),'trials':[]}
        for cfg in configs:
            y,res=linear(x,w,r,s,cfg,resources=True);assert bits(y,ref),(k,cfg)
            row['trials'].append({'config':cfg,'bits_equal':True,'resources':res,'ms':do_bench_cudagraph(lambda:linear(x,w,r,s,cfg),rep=20)})
        report['records'].append(row);Path('results/residual_gemv_async_probe.json').write_text(json.dumps(report,indent=2)+'\n')
        print(k,'current',row['current_ms'],'trials',len(row['trials']),'best',sorted(row['trials'],key=lambda r:r['ms'])[:3],flush=True)
    report['all_exact']=True;Path('results/residual_gemv_async_probe.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

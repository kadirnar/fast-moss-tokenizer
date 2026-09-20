"""Retune the two CUDA residual epilogues that lose with matrix-only schedules."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.short_ffn import CONFIGS,linear as fused
from benchmarks.native_layer_norm_tune import bits
from fast_moss.wide_matrices import linear as current_matrix
from fast_moss.kernels import scale_add
from fast_moss.normalization import compiler
from fast_moss.ffn import math_library
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(521);bindings=compiler();library=math_library();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'epilogue-aware CUDA scheduling search; exploratory warm graph timing','previous_commit':'c67e3a9','revision':REVISION,'records':[]}
    for shape in [(3,1280,5120),(8,768,3072)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];m,n,k=shape
        r=torch.randn(m,n,device='cuda');s=torch.randn(n,device='cuda')*.01;ref=r+F.linear(x,w)*s
        baseline=lambda:scale_add(r,current_matrix(x,w,bindings),s)
        record={'shape':shape,'original_config':CONFIGS[shape][1],'current_ms':do_bench_cudagraph(baseline,rep=30),'trials':[]}
        parts=k//256
        for rows in ((2,3,4) if m==3 else (4,6,8)):
            for cols in (1,2,4,8):
                for groups in (2,4,parts):
                    if cols*groups*16>1024 or parts*rows*cols*64>49152:continue
                    for u in (4,16):
                        for distribute in (False,True):
                            config=(rows,cols,groups,u,distribute);CONFIGS[shape]=('wide',config)
                            fn=lambda:fused(x,w,'residual',r,s,library,bindings)
                            out,resources=fused(x,w,'residual',r,s,library,bindings,True);exact=bits(ref,out)
                            record['trials'].append({'config':config,'resources':resources,'bits_equal':exact,'ms':do_bench_cudagraph(fn,rep=12) if exact else None})
        valid=[t for t in record['trials'] if t['bits_equal']];print(shape,'current',record['current_ms'],'best',min(valid,key=lambda t:t['ms']),'exact',len(valid),'/',len(record['trials']),flush=True)
        report['records'].append(record);Path('results/short_ffn_retune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

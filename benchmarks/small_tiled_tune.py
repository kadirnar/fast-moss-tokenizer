"""Research-only exact small GEMM tile timings."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.small_tiled_arithmetic import tiled
from benchmarks.compare import difference
from fast_moss.loading import strict_precision

@torch.inference_mode()
def main():
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only warm component timings; not model speedups','records':[]}
    for shape,case in cases.items():
        m,n,k=shape
        if m not in (3,6,12) or n not in (1280,3840,5120,768,2304,3072):continue
        x,w=case['x'],case['weight'];ref=F.linear(x,w)
        r={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=40),'trials':[]}
        for bn in (4,8,16,32):
            for warps in (1,2,4,8):
                fn=lambda:tiled(x,w,bn=bn,warps=warps)
                d=difference(ref,fn())
                ms=do_bench_cudagraph(fn,rep=30)
                r['trials'].append({'bn':bn,'warps':warps,'difference':d,'ms':ms})
        report['records'].append(r)
        best=min(r['trials'],key=lambda t:t['ms'])
        print(shape,'native',r['native_ms'],'best',best,flush=True)
        Path('results/small_tiled_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

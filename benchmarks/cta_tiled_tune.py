"""Single-CTA split-K search on the remaining native transformer shapes."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.cta_tiled_matrix import linear,compile_kernel
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision,REVISION

SHAPES=[(4,3072,768),(8,2304,768),(4,2304,768),(4,768,768),
        (6,3072,768),(6,2304,768),(12,2304,768),(12,768,768)]

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'single-launch shared-memory K partitions, exact native tile/lane order; warm exploratory timings',
        'previous_commit':'c2e4722','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape in SHAPES:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        record={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=30),'trials':[]}
        m=shape[0]
        for rows in sorted({2,4,6,m}):
            for cols in (1,2,4,8):
                for unroll in (4,16):
                    for distribute in (False,True):
                        config=(rows,cols,unroll,distribute);out=linear(x,w,config)
                        record['trials'].append({'config':config,'bits_equal':bits(ref,out),'resources':compile_kernel(shape,config)[2],
                            'ms':do_bench_cudagraph(lambda:linear(x,w,config),rep=12)})
        best=min(record['trials'],key=lambda r:r['ms'])
        print(shape,'native',record['native_ms'],'best',best,'all_exact',all(t['bits_equal'] for t in record['trials']),flush=True)
        report['records'].append(record);Path('results/cta_tiled_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

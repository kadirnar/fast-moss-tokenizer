"""Replace global partial buffers for long-K small matrices, preserving arithmetic."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.wide_cta_matrix import linear,compile_kernel,SOURCE
from benchmarks.native_layer_norm_tune import bits
from fast_moss.small_matrices import linear as current,CONFIGS
from fast_moss.loading import strict_precision,REVISION
SHAPES=[s for s,c in CONFIGS.items() if (s[0]>1 and s[2]>=3072) or c[0]=='split']

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research long-K single-block shared partials; exploratory warm timings',
        'previous_commit':'397db43','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for shape in SHAPES:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        assert bits(current(x,w),ref)
        record={'shape':shape,'current_config':CONFIGS[shape],'current_ms':do_bench_cudagraph(lambda:current(x,w),rep=30),'trials':[]}
        m,n,k=shape;parts=k//256
        for rows in sorted({2,4,min(m,8)}):
            for cols in (1,2,4,8):
                for groups in sorted({2,4,parts}):
                    if groups>parts or 16*cols*groups>1024 or parts*rows*cols*64>49152:continue
                    for unroll in (4,16):
                        for distribute in (False,True):
                            config=(rows,cols,groups,unroll,distribute);out=linear(x,w,config)
                            exact=bits(ref,out)
                            record['trials'].append({'config':config,'bits_equal':exact,'resources':compile_kernel(shape,config)[2],
                                'ms':do_bench_cudagraph(lambda:linear(x,w,config),rep=12) if exact else None})
        valid=[r for r in record['trials'] if r['bits_equal']]
        best=min(valid,key=lambda r:r['ms']) if valid else None
        print(shape,'current',record['current_ms'],'best',best,'exact',len(valid),'/',len(record['trials']),flush=True)
        report['records'].append(record);Path('results/wide_cta_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

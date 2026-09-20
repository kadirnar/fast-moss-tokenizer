"""Tune exact one-row cyclic GEMV scheduling on captured checkpoint matrices."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.gemv_order import gemv
from benchmarks.compare import difference
from benchmarks.ffn_resources import resources
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    order=json.loads(Path('results/gemv_order.json').read_text())
    report={'scope':'exact full-width one-row GEMV launch/unroll search, warm component timings',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for record in order['records']:
        shape=tuple(record['shape']);x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        lanes=next(t['lanes'] for t in record['trials'] if t['bits_equal'])
        row={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=25),'trials':[]}
        configs=[(lanes,bn,warps,unroll) for bn in [4,8,16,32] for warps in [1,2,4]
                 for unroll in [4,16,32] if bn*lanes>=32*warps]
        configs.append((lanes,8,4,1))
        for config in configs:
            fn=lambda:gemv(x,w,*config,zero=True)
            out,kernel=gemv(x,w,*config,zero=True,return_kernel=True)
            d=difference(ref,out);bits=torch.equal(ref.view(torch.int32),out.view(torch.int32))
            ms=do_bench_cudagraph(fn,rep=15)
            row['trials'].append({'config':config,'difference':d,'bits_equal':bits,'ms':ms,'resources':resources(kernel)})
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['ms'])
        print(shape,'native',row['native_ms'],'best',best,flush=True)
        Path('results/gemv_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

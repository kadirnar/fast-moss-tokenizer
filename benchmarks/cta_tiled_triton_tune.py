"""Compare a repo-native Triton implementation of the concurrent K-tile idea."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.cta_tiled_triton import linear
from benchmarks.cta_tiled_tune import SHAPES
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'Triton concurrent native K-tile warm search','previous_commit':'c2e4722','revision':REVISION,'records':[]}
    for shape in SHAPES:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        row={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=20),'trials':[]}
        for bm in (4,8):
            for bn in (2,4,8):
                for warps in (2,4,8):
                    for u in (4,16):
                        cfg=(bm,bn,warps,u);out,res=linear(x,w,cfg,True)
                        row['trials'].append({'config':cfg,'bits_equal':bits(out,ref),'resources':res,
                            'ms':do_bench_cudagraph(lambda:linear(x,w,cfg),rep=12)})
        report['records'].append(row);print(shape,'native',row['native_ms'],'best',min(row['trials'],key=lambda r:r['ms']),flush=True)
        Path('results/cta_tiled_triton_tune.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

"""Isolate libdevice erf equivalence before accepting a fused GELU."""
import json
import hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from benchmarks.ordered_epilogue import mul_rn, add_rn
from benchmarks.compare import difference
from fast_moss.ffn import math_library


@triton.jit
def _math(X,Y,N:tl.constexpr,GELU:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    x=tl.load(X+i,i<N,0)
    if GELU:
        y=mul_rn(mul_rn(x,0.5),add_rn(libdevice.erf(mul_rn(x,0.7071067811865476)),1.0))
    else:
        y=libdevice.erf(x)
    tl.store(Y+i,y,i<N)


@torch.inference_mode()
def main():
    torch.manual_seed(857)
    x=torch.cat([torch.linspace(-10,10,1048576,device='cuda'),torch.randn(1048576,device='cuda')])
    report={'scope':'same-input erf/GELU library isolation','torch':torch.__version__,'triton':triton.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    report['torch_composed_gelu']=difference(torch.nn.functional.gelu(x),x*.5*(1+torch.erf(x*0.7071067811865476)))
    default=Path(triton.__file__).parent/'backends/nvidia/lib/libdevice.10.bc'
    for path in [default,Path(math_library())]:
        for gelu in [False,True]:
            ref=torch.nn.functional.gelu(x) if gelu else torch.erf(x)
            for fusion in [False,True]:
                y=torch.empty_like(x)
                _math[(triton.cdiv(x.numel(),256),)](x,y,x.numel(),gelu,enable_fp_fusion=fusion,
                                                   extern_libs={'libdevice':str(path)})
                r={'libdevice':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'gelu':gelu,'fusion':fusion,'difference':difference(ref,y)}
                report['records'].append(r);print(r,flush=True)
    Path('results/gelu_math.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

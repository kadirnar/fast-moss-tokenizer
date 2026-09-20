"""Inspect the two integrated native FFN GEMV kernels and verify exact outputs."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from fast_moss.ffn import _ffn_gemv,gemv_linear,math_library,LIBDEVICE_SHA256
from fast_moss.small_matrices import CONFIGS
from fast_moss.loading import strict_precision,REVISION


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(3269);library=math_library()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'integrated native FP32 FFN GEMV compiler resources and captured operands',
            'revision':REVISION,'libdevice_sha256':LIBDEVICE_SHA256,'records':[]}
    for shape,mode in [((1,5120,1280),'gelu'),((1,1280,5120),'residual')]:
        x,w=cases[shape]['x'],cases[shape]['weight'];_,n,k=shape
        r=torch.randn(1,n,device='cuda');s=torch.randn(n,device='cuda')*.01
        y=torch.empty((1,n),device='cuda');_,lanes,bn,warps,u=CONFIGS[shape]
        kernel=_ffn_gemv[((n+bn-1)//bn,)](x,w,r,s,y,n,k,lanes,bn,u,mode,
            num_warps=warps,enable_fp_fusion=False,extern_libs={'libdevice':library})
        raw=F.linear(x,w);ref=F.gelu(raw) if mode=='gelu' else r+raw*s
        helper=gemv_linear(x,w,mode,r,s,library)
        ptx=kernel.asm['ptx']
        record={'shape':shape,'mode':mode,'config':CONFIGS[shape],
            'kernel_bits_equal':torch.equal(ref.view(torch.int32),y.view(torch.int32)),
            'helper_bits_equal':torch.equal(ref.view(torch.int32),helper.view(torch.int32)),
            'registers':kernel.n_regs,'spills':kernel.n_spills,'shared_bytes':kernel.metadata.shared,
            'fp32_fma_instructions':ptx.count('fma.rn.f32'),
            'matrix_instructions':sum(ptx.count(s) for s in ('mma.sync','wgmma.','tcgen05.mma'))}
        report['records'].append(record)
    Path('results/gemv_epilogue_resources.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(r['kernel_bits_equal'] and r['helper_bits_equal'] and r['spills']==r['matrix_instructions']==0 for r in report['records'])
    print(report)

if __name__=='__main__':main()

"""Inspect the compiled runtime FFN kernels and reconfirm actual-input results."""
import json
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from fast_moss.loading import strict_precision
from fast_moss.ordered_matrices import _partials, _reduce
from fast_moss.ffn import _ffn_reduce, math_library


def resources(kernel):
    ptx = kernel.asm['ptx']
    return {'registers': kernel.n_regs, 'spills': kernel.n_spills,
            'shared_bytes': kernel.metadata.shared,
            'ptx_fma_rn_f32': ptx.count('fma.rn.f32'),
            'ptx_mma_instructions': len(re.findall(r'\b(?:mma\.|wgmma\.|tcgen05\.mma)', ptx))}


@torch.inference_mode()
def main():
    strict_precision()
    torch.manual_seed(747)
    library = math_library()
    cases = torch.load('results/matrix_inputs.pt', weights_only=True)
    report = {'scope': 'compiled runtime FFN PTX/resources and actual-input equality',
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'records': []}
    for shape, mode in [((24,5120,1280),'gelu'), ((24,1280,5120),'residual')]:
        m,n,k = shape
        x,w = cases[shape]['x'],cases[shape]['weight']
        packed = w.T.contiguous()
        p = torch.empty((k//256,m,n),device=x.device)
        out = torch.empty((m,n),device=x.device)
        residual = torch.randn_like(out)
        scale = torch.randn(n,device=x.device)
        ref = F.linear(x,w)
        for stages in [3,2]:
            main_kernel = _partials[(1,n//128,k//256)](
                x,packed,p,m,n,k,256,32,128,32,num_warps=4,
                num_stages=stages,enable_fp_fusion=False)
            _reduce[(m*n//256,)](p,out,m*n,k//256,256,enable_fp_fusion=False)
            matrix_exact = torch.equal(ref.view(torch.int32),out.view(torch.int32))
            reduce_kernel = _ffn_reduce[(m*n//256,)](
                p,residual,scale,out,m*n,n,k//256,mode,256,
                enable_fp_fusion=False,extern_libs={'libdevice':library})
            expected = F.gelu(ref) if mode == 'gelu' else residual+ref*scale
            report['records'].append({'shape':shape,'stages':stages,'mode':mode,
                'main':resources(main_kernel),'fused_reduce':resources(reduce_kernel),
                'matrix_bits_equal':matrix_exact,
                'fused_bits_equal':torch.equal(expected.view(torch.int32),out.view(torch.int32))})
    report['all_exact'] = all(r['matrix_bits_equal'] and r['fused_bits_equal'] for r in report['records'])
    Path('results/ffn_kernel_resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['all_exact']:
        raise SystemExit('FFN resource fidelity gate failed')


if __name__ == '__main__':
    main()

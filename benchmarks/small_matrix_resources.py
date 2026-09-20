"""Compiled resources and actual-weight bits for accepted small-row matrices."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from fast_moss.loading import strict_precision
from fast_moss.small_matrices import CONFIGS,linear,_small_gemv,_small_fixed,_small_grouped,_small_parts,_small_reduce
from benchmarks.ffn_resources import resources

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/small_matrix_resources.json')
    args=parser.parse_args()
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'compiled native-layout small matrix resources and bit equality',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape,config in CONFIGS.items():
        m,n,k=shape;strategy,bm,bn,warps,unroll=config
        x,w=cases[shape]['x'],cases[shape]['weight'];out=torch.empty((m,n),device='cuda')
        if strategy=='gemv':
            kernels=[_small_gemv[(triton.cdiv(n,bn),)](
                x,w,out,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)]
            scratch=0
        elif strategy=='fixed':
            kernels=[_small_fixed[(triton.cdiv(n,bn),triton.cdiv(m,bm))](
                x,w,out,m,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)]
            scratch=0
        elif strategy=='grouped':
            kernels=[_small_grouped[(triton.cdiv(n,bn),triton.cdiv(m,bm))](
                x,w,out,m,n,k,bm,bn,unroll,num_warps=warps,enable_fp_fusion=False)]
            scratch=0
        else:
            parts=triton.cdiv(k,256);p=torch.empty((parts,m,n,16),device='cuda')
            kernels=[_small_parts[(triton.cdiv(n,bn),triton.cdiv(m,bm),parts)](
                x,w,p,m,n,k,bm,bn,num_warps=warps,enable_fp_fusion=False),
                _small_reduce[(triton.cdiv(m*n,32),)](p,out,m*n,parts,32,num_warps=4,enable_fp_fusion=False)]
            scratch=p.numel()*p.element_size()
        expected=F.linear(x,w);actual=linear(x,w)
        report['records'].append({'shape':shape,'config':config,'kernels':[resources(kernel) for kernel in kernels],
            'scratch_bytes':scratch,'probe_bits_equal':torch.equal(expected.view(torch.int32),out.view(torch.int32)),
            'runtime_bits_equal':torch.equal(expected.view(torch.int32),actual.view(torch.int32))})
    report['all_exact']=all(r['probe_bits_equal'] and r['runtime_bits_equal'] for r in report['records'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('all_exact',report['all_exact'])
    if not report['all_exact']:raise SystemExit('Small matrix resource gate failed')

if __name__=='__main__':main()

"""Inspect compiled runtime resources and arithmetic for every ordered shape."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from fast_moss.loading import strict_precision
from fast_moss.ordered_matrices import CONFIGS, PRESERVE_SIGNED_ZERO, linear as runtime_linear
from benchmarks.ordered_shapes_confirm import linear
from benchmarks.ffn_resources import resources


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/ordered_shapes_resources.json')
    args=parser.parse_args()
    strict_precision()
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'all compiled ordered runtime shapes, PTX/resources and actual-input equality',
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'records':[]}
    for shape,config in sorted(CONFIGS.items()):
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        out,kernel=linear(x,packed,config,return_kernel=True,exact_zero=True)
        ref=F.linear(x,w);runtime=runtime_linear(x,packed)
        report['records'].append({'shape':shape,'config':config,'main':resources(kernel),
            'preserve_signed_zero':shape in PRESERVE_SIGNED_ZERO,
            'exact_tail':shape in PRESERVE_SIGNED_ZERO and shape[2]%config[0]!=0,
            'scratch_bytes':((shape[2]+config[0]-1)//config[0])*shape[0]*shape[1]*4,
            'probe_bits_equal':torch.equal(ref.view(torch.int32),out.view(torch.int32)),
            'runtime_bits_equal':torch.equal(ref.view(torch.int32),runtime.view(torch.int32))})
    report['all_exact']=all(r['probe_bits_equal'] and r['runtime_bits_equal'] for r in report['records'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print('all_exact',report['all_exact'],'shapes',len(report['records']))
    if not report['all_exact']:raise SystemExit('Ordered runtime resource gate failed')


if __name__=='__main__':main()

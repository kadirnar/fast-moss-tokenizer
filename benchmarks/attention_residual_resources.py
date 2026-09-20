"""Re-audit compiler/output provenance after canonical singleton allocation fix."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks.attention_residual import SOURCE,linear
from fast_moss.loading import strict_precision
from fast_moss.ffn import _ffn_gemv,_mul,_add
from fast_moss.strided_ffn import _strided_ffn_fixed,_epilogue

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',action='store_true');args=parser.parse_args()
    helper=linear
    if args.runtime:
        from fast_moss.attention_residual import linear as helper,SOURCE as runtime_source
        assert runtime_source==SOURCE
    strict_precision();torch.manual_seed(972)
    captured=torch.load('results/matrix_inputs.pt',weights_only=True)
    prior=json.loads(Path('results/attention_residual_probe.json').read_text())
    report={'scope':'post-allocation-fix compiler/output audit against component-probe binaries','previous_commit':'84caafc','integration_parent':'f104647','runtime':args.runtime,
        'cuda_source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),
        'triton_source_sha256':{name:hashlib.sha256(fn.src.encode()).hexdigest() for name,fn in [('_ffn_gemv',_ffn_gemv),('_strided_ffn_fixed',_strided_ffn_fixed),('_epilogue',_epilogue),('_mul',_mul),('_add',_add)]},'records':[]}
    assert report['cuda_source_sha256']==prior['cuda_source_sha256']
    for row in prior['records']:
        shape=tuple(row['shape']);m,n,k=shape;t=row['time'];x,w=[captured[shape][name] for name in ('x','weight')]
        residual=torch.randn(m//t,n,t,device='cuda').transpose(1,2);scale=torch.randn(n,device='cuda')*.1
        out,res=helper(x,w,residual,scale,True);ref=residual+F.linear(x,w).reshape_as(residual)*scale
        assert torch.equal(out.view(torch.int32),ref.view(torch.int32)) and out.stride()==ref.stride()
        assert res==row['resources']
        report['records'].append({'shape':shape,'time':t,'bits_equal':True,'stride_equal':True,'resources_equal_to_probe':True,**res})
    records=[]
    if not args.runtime:
        package=json.loads(Path('results/norm_projection_package.json').read_text())
        for row in package['records']:
            digest=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest();assert digest==row['sha256']
            records.append({'path':row['path'],'sha256':digest,'matches_verified_package':True})
        report['production_files']=records;report['production_unchanged']=True
    name='attention_residual_runtime_resources' if args.runtime else 'attention_residual_resources'
    Path('results',name+'.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['records']),'production compiler variants matching research' if args.runtime else 'research compiler variants and unchanged production files')

if __name__=='__main__':main()

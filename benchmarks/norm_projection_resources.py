"""Audit selected production normalization/projection arithmetic and compiler output."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from fast_moss import norm_projection as runtime
from benchmarks import norm_gemv as research
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision()
    matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')]
    assert runtime.SOURCE==research.SOURCE
    prior=json.loads(Path('results/norm_gemv_resources.json').read_text())
    report={'scope':'selected production normalization/projection compiler and captured-operand audit',
        'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'research_source_equal':True,'source_sha256':hashlib.sha256(runtime.SOURCE.encode()).hexdigest(),
        'flags':['--gpu-architecture=sm_120','--std=c++17','--ftz=false','--fmad=true'],'records':[]}
    for mode,cfg in runtime.CONFIGS.items():
        n=3840 if mode=='none' else 5120;w=matrices[(1,n,1280)]['weight']
        normalized=F.layer_norm(x,(1280,),g,b,eps);ref=F.linear(normalized,w)
        if mode=='gelu':ref=F.gelu(ref)
        out,res=runtime.linear(x,w,g,b,eps,mode,resources=True);assert bits(out,ref)
        _,_,_,blob=runtime.compile_kernel(n,mode,cfg)
        digest=hashlib.sha256(blob).hexdigest()
        original=next(r for r in prior['records'] if r['mode']==mode and r['config']==list(cfg))
        assert digest==original['cubin_sha256']
        assert res['local_bytes']==res['local_loads']==res['local_stores']==res['matrix_instructions']==0
        report['records'].append({'mode':mode,'config':cfg,'bits_equal':True,**res,'cubin_sha256':digest,'research_cubin_equal':True})
    Path('results/norm_projection_resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['records']),'production variants',flush=True)

if __name__=='__main__':main()

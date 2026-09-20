"""Compiler and source provenance for the confirmed fused norm/GEMV variants."""
import hashlib,json
from pathlib import Path
import torch
import torch.nn.functional as F
from benchmarks import norm_gemv as base,norm_gemv_cache as cache
from benchmarks.native_layer_norm_tune import bits
from fast_moss.normalization import SOURCE as NORMALIZATION_SOURCE
from fast_moss.loading import strict_precision,REVISION

@torch.inference_mode()
def main():
    strict_precision();matrices=torch.load('results/matrix_inputs.pt',weights_only=True)
    norm=torch.load('results/layer_norm_inputs.pt',weights_only=True)[(1,1280)]
    x,g,b,eps=[norm[k] for k in ('x','weight','bias','eps')]
    confirm=json.loads(Path('results/norm_gemv_confirm.json').read_text());policies=json.loads(Path('results/norm_gemv_cache_probe.json').read_text())
    report={'scope':'research norm/GEMV compiler resources, original-operand arithmetic and source provenance',
        'previous_commit':'b240bd2','revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':{'base':hashlib.sha256(base.SOURCE.encode()).hexdigest(),'cache':hashlib.sha256(cache.SOURCE.encode()).hexdigest(),
            'normalization':hashlib.sha256(NORMALIZATION_SOURCE.encode()).hexdigest()},
        'flags':['--gpu-architecture=sm_120','--std=c++17','--ftz=false','--fmad=true'],'records':[]}
    for row,policy in zip(confirm['records'],policies['records']):
        _,n,k=row['shape'];mode=row['mode'];w=matrices[(1,n,k)]['weight'];normalized=F.layer_norm(x,(1280,),g,b,eps)
        ref=F.linear(normalized,w);ref=F.gelu(ref) if mode=='gelu' else ref
        for cfg in [tuple(c) for c in row['configs']]+[tuple(json.loads(name)) for name in policy['resources']]:
            module=cache if len(cfg)==6 else base
            out,res=module.linear(x,w,g,b,eps,mode,cfg,resources=True);assert bits(out,ref)
            _,_,_,blob=module.compile_kernel(n,mode,cfg)
            assert res['local_bytes']==res['local_loads']==res['local_stores']==res['matrix_instructions']==0
            if module is cache:assert res['cache_load_instructions']>0
            report['records'].append({'shape':row['shape'],'mode':mode,'config':cfg,'bits_equal':True,**res,'cubin_sha256':hashlib.sha256(blob).hexdigest()})
    Path('results/norm_gemv_resources.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(report['records']),'compiled variants')

if __name__=='__main__':main()

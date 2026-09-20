"""Audit production strided-FFN compiler resources, arithmetic and source provenance."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
from fast_moss.strided_ffn import CONFIGS,linear,cuda_source,compile_kernel,_strided_ffn_fixed,_epilogue
from fast_moss.ffn import math_library,_mul,_add
from fast_moss.normalization import compiler,_check
from fast_moss.loading import strict_precision,REVISION
from benchmarks.native_layer_norm_tune import bits

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(4419);cases=torch.load('results/matrix_inputs.pt',weights_only=True);bindings=compiler();cu,nvrtc=bindings;library=math_library()
    report={'scope':'production strided FFN resources and captured arithmetic audit','revision':REVISION,'torch':torch.__version__,'triton':triton.__version__,
        'gpu':torch.cuda.get_device_name(),'library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        'triton_source_sha256':{name:hashlib.sha256(fn.src.encode()).hexdigest() for name,fn in [('_strided_ffn_fixed',_strided_ffn_fixed),('_epilogue',_epilogue),('_mul',_mul),('_add',_add)]},
        'cuda_source_sha256':{family:hashlib.sha256(cuda_source(family).encode()).hexdigest() for family in ('cuda','wide')},'records':[]}
    geometries=[(shape,b,shape[0]//b) for shape in sorted(CONFIGS) if shape[1]<shape[2] for b in range(1,shape[0]) if shape[0]%b==0]
    for shape,batch,time in geometries:
        family,config=CONFIGS[shape];m,n,k=shape;mode='residual';x,w=cases[shape]['x'],cases[shape]['weight']
        residual=torch.randn(batch,n,time,device='cuda').transpose(1,2);scale=torch.randn(n,device='cuda')*.01
        out,resources=linear(x,w,residual,scale,library,bindings,True)
        raw=F.linear(x,w).reshape_as(residual);ref=residual+raw*scale
        assert bits(out,ref)
        record={'batch':batch,'time':time,'shape':shape,'family':family,'config':config,'mode':mode,'bits_equal':True,**resources}
        if family=='fixed':
            rows,cols,warps,u=config
            kernel=_strided_ffn_fixed[(triton.cdiv(n,cols),triton.cdiv(m,rows))](x,w,residual,scale,out,m,n,k,rows,cols,u,mode,time,num_warps=warps,enable_fp_fusion=False,extern_libs={'libdevice':library})
            ptx=kernel.asm['ptx'];record['spills']=kernel.n_spills;record['flags']={'enable_fp_fusion':False,'num_warps':warps}
            assert bits(out,ref)
        else:
            if family=='wide':rows,cols,groups,u,distribute=config
            else:rows,cols,u,distribute=config;groups=k//256
            definitions=dict(T=time,M=m,N=n,K=k,ROWS=rows,COLS=cols,GROUPS=groups,UNROLL=u,DISTRIBUTE=int(distribute),MODE=int(mode=='residual'))
            source=('\n'.join(f'#define {a} {b}' for a,b in definitions.items())+'\n'+cuda_source(family)).encode()
            program=_check(nvrtc.nvrtcCreateProgram(source,b'short_ffn.cu',0,[],[]));options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
            try:
                _check(nvrtc.nvrtcCompileProgram(program,len(options),options))
                ptx=b' '*_check(nvrtc.nvrtcGetPTXSize(program));_check(nvrtc.nvrtcGetPTX(program,ptx));ptx=ptx.decode()
                blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
            finally:_check(nvrtc.nvrtcDestroyProgram(program))
            record['cubin_sha256']=hashlib.sha256(blob).hexdigest();record['flags']=[x.decode() for x in options]
        record.update(fp32_fma_instructions=ptx.count('fma.rn.f32'),local_loads=ptx.count('ld.local'),local_stores=ptx.count('st.local'),
                      matrix_instructions=sum(ptx.count(s) for s in ('mma.sync','wgmma.','tcgen05.mma')))
        assert record['local_bytes']==record['matrix_instructions']==record['local_loads']==record['local_stores']==0
        report['records'].append(record)
    Path('results/strided_ffn_resources.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

if __name__=='__main__':main()

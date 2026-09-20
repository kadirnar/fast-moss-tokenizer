"""Check NVRTC GELU against native PyTorch before using it in CUDA epilogues."""
import ctypes
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from fast_moss.normalization import compiler,_check
from fast_moss.loading import strict_precision

SOURCE=r'''
__device__ __forceinline__ float add(float a,float b){float y;asm("add.rn.f32 %0, %1, %2;":"=f"(y):"f"(a),"f"(b));return y;}
__device__ __forceinline__ float mul(float a,float b){float y;asm("mul.rn.f32 %0, %1, %2;":"=f"(y):"f"(a),"f"(b));return y;}
extern "C" __global__ void math_probe(const float* X,float* Y,int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N){float x=X[i];Y[i]=mul(mul(x,.5f),add(erff(mul(x,.7071067811865476f)),1.f));}
}
'''

@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(9127)
    special=torch.tensor([0.,-0.,1.401298464324817e-45,-1.401298464324817e-45,1e-38,-1e-38,1e20,-1e20,float('inf'),-float('inf'),float('nan')],device='cuda')
    x=torch.cat([torch.linspace(-10,10,1048576,device='cuda'),torch.randn(1048576,device='cuda'),special]);ref=F.gelu(x)
    cu,nvrtc=compiler();report={'scope':'NVRTC GELU math isolation, explicit outer FP32 rounding','source_sha256':hashlib.sha256(SOURCE.encode()).hexdigest(),'records':[]}
    for fusion in ('false','true'):
        program=_check(nvrtc.nvrtcCreateProgram(SOURCE.encode(),b'gelu_probe.cu',0,[],[]));options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',f'--fmad={fusion}'.encode()]
        try:
            result=nvrtc.nvrtcCompileProgram(program,len(options),options)
            if int(result[0]):
                log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
            blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
        finally:_check(nvrtc.nvrtcDestroyProgram(program))
        module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'math_probe'));y=torch.empty_like(x)
        values=[ctypes.c_void_p(x.data_ptr()),ctypes.c_void_p(y.data_ptr()),ctypes.c_int(x.numel())]
        pointers=(ctypes.c_void_p*3)(*(ctypes.addressof(v) for v in values))
        _check(cu.cuLaunchKernel(fn,(x.numel()+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
        torch.cuda.synchronize();diff=ref.view(torch.int32)!=y.view(torch.int32)
        r={'fmad':fusion,'elements':x.numel(),'different_bits':int(diff.sum()),'finite_different_bits':int(diff[torch.isfinite(ref)].sum()),'special_reference_bits':ref[-special.numel():].view(torch.int32).tolist(),'special_candidate_bits':y[-special.numel():].view(torch.int32).tolist()}
        report['records'].append(r);print(r,flush=True);_check(cu.cuModuleUnload(module))
    Path('results/short_ffn_cuda_math.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

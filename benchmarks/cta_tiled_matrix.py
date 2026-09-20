"""Research single-launch split-K with native tile/lane accumulation order."""
import ctypes
import torch
from cuda.bindings import driver as cu,nvrtc
from fast_moss.normalization import _check

SOURCE=r'''
__device__ __forceinline__ float add(float a,float b){
    float value;asm("add.rn.f32 %0, %1, %2;":"=f"(value):"f"(a),"f"(b));return value;
}
extern "C" __global__ void cta_tiled(const float* X,const float* W,float* Y){
    constexpr int PARTS=K/256;
    int lane=threadIdx.x%16;
    int output=(threadIdx.x/16)%COLS;
    int part=threadIdx.x/(16*COLS);
    int n=blockIdx.x*COLS+output,m=blockIdx.y*ROWS;
    float acc[ROWS]={};
    #pragma unroll UNROLL
    for(int step=0;step<16;++step){
        int k=part*256+step*16+lane;
        float w=n<N?W[n*K+k]:0.f;
        #pragma unroll
        for(int r=0;r<ROWS;++r){
            float x=m+r<M?X[(m+r)*K+k]:0.f;
            acc[r]=__fmaf_rn(x,w,acc[r]);
        }
    }
    __shared__ float partial[PARTS*ROWS*COLS*16];
    #pragma unroll
    for(int r=0;r<ROWS;++r)partial[((part*ROWS+r)*COLS+output)*16+lane]=acc[r];
    __syncthreads();
    // Every output group has exactly 16 participating lanes, including
    // the half-warp at the end of a 48-thread block. Do not sample activemask
    // inside the divergent reducer branch under independent scheduling.
    unsigned mask=0xffffu << (threadIdx.x%32/16*16);
    #pragma unroll
    for(int r=0;r<ROWS;++r){
        if(part==(DISTRIBUTE ? r%PARTS : 0)){
            float sum=0.f;
            #pragma unroll
            for(int p=0;p<PARTS;++p)sum=add(sum,partial[((p*ROWS+r)*COLS+output)*16+lane]);
            float value=__shfl_sync(mask,sum,0,16);
            #pragma unroll
            for(int l=1;l<16;++l)value=add(value,__shfl_sync(mask,sum,l,16));
            if(lane==0 && m+r<M && n<N)Y[(m+r)*N+n]=value;
        }
    }
}
'''
CACHE={}


def compile_kernel(shape,config):
    key=(torch.cuda.current_device(),shape,config)
    if key in CACHE:return CACHE[key]
    m,n,k=shape;rows,cols,unroll,distribute=config
    defs=dict(M=m,N=n,K=k,ROWS=rows,COLS=cols,UNROLL=unroll,DISTRIBUTE=int(distribute))
    source=('\n'.join(f'#define {a} {b}' for a,b in defs.items())+'\n'+SOURCE).encode()
    program=_check(nvrtc.nvrtcCreateProgram(source,b'cta_tiled.cu',0,[],[]))
    try:
        opts=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
        result=nvrtc.nvrtcCompileProgram(program,len(opts),opts)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'cta_tiled'))
    a=cu.CUfunction_attribute
    resources={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [
        ('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    CACHE[key]=(module,fn,resources);return CACHE[key]


def linear(x,w,config):
    rows,cols,unroll,distribute=config
    if (x.ndim!=2 or w.ndim!=2 or x.shape[1]!=768 or w.shape[1]!=768
        or not x.is_cuda or x.dtype!=torch.float32 or w.dtype!=x.dtype or w.device!=x.device
        or not x.is_contiguous() or not w.is_contiguous() or x.shape[0]<1 or w.shape[0]<1
        or rows not in (1,2,3,4,6,8,12) or cols not in (1,2,4,8) or unroll not in (1,4,16)):
        raise ValueError('Expected contiguous CUDA FP32 K=768 operands and validated launch dimensions')
    m,k=x.shape;n=w.shape[0]
    with torch.cuda.device(x.device):
        _,fn,_=compile_kernel((m,n,k),tuple(config));out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        vals=[ctypes.c_void_p(v.data_ptr()) for v in (x,w,out)];pointers=(ctypes.c_void_p*3)(*(ctypes.addressof(v) for v in vals))
        _check(cu.cuLaunchKernel(fn,(n+cols-1)//cols,(m+rows-1)//rows,1,16*cols*(k//256),1,1,0,
            cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return out

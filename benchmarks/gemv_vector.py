"""Exact native-layout CUDA GEMV with vector loads and ordered prefetch windows."""
import ctypes
import torch
from cuda.bindings import driver as cu,nvrtc
from benchmarks.small_vector_cuda import check

SOURCE=r'''
__device__ __forceinline__ void fetch(const float* p,float (&v)[VEC]){
  #if VEC==1
  v[0]=*p;
  #elif VEC==2
  float2 q=*reinterpret_cast<const float2*>(p);v[0]=q.x;v[1]=q.y;
  #else
  #pragma unroll
  for(int i=0;i<VEC;i+=4){
    float4 q=*reinterpret_cast<const float4*>(p+i);
    v[i]=q.x;v[i+1]=q.y;v[i+2]=q.z;v[i+3]=q.w;
  }
  #endif
}
extern "C" __global__ void vector_gemv(const float* __restrict__ X,
                                      const float* __restrict__ W,float* Y){
  constexpr int THREAD_LANES=LANES/VEC;
  int lane=threadIdx.x%THREAD_LANES;
  int n=blockIdx.x*(THREADS/THREAD_LANES)+threadIdx.x/THREAD_LANES;
  float acc[VEC]={};
  #pragma unroll UNROLL
  for(int start=0;start<K/LANES;start+=PREFETCH){
    float xs[PREFETCH][VEC],ws[PREFETCH][VEC]={};
    #pragma unroll
    for(int p=0;p<PREFETCH;++p){
      int k=(start+p)*LANES+lane*VEC;
      fetch(X+k,xs[p]);
      if(n<N)fetch(W+n*K+k,ws[p]);
    }
    #pragma unroll
    for(int p=0;p<PREFETCH;++p){
      #pragma unroll
      for(int v=0;v<VEC;++v)acc[v]=__fmaf_rn(xs[p][v],ws[p][v],acc[v]);
    }
  }
  #pragma unroll
  for(int delta=LANES/2;delta>=VEC;delta/=2){
    #pragma unroll
    for(int v=0;v<VEC;++v)
      acc[v]=__fadd_rn(acc[v],__shfl_xor_sync(0xffffffff,acc[v],delta/VEC,THREAD_LANES));
  }
  #pragma unroll
  for(int delta=VEC/2;delta>0;delta/=2){
    float next[VEC];
    #pragma unroll
    for(int v=0;v<VEC;++v)next[v]=__fadd_rn(acc[v],acc[v^delta]);
    #pragma unroll
    for(int v=0;v<VEC;++v)acc[v]=next[v];
  }
  if(lane==0 && n<N)Y[n]=__fadd_rn(acc[0],0.f);
}
'''
CACHE={}


def compile_kernel(n,k,config):
    lanes,vec,threads,prefetch,u=config;key=(n,k,tuple(config))
    if key in CACHE:return CACHE[key]
    defs=dict(N=n,K=k,LANES=lanes,VEC=vec,THREADS=threads,PREFETCH=prefetch,UNROLL=u)
    source=('\n'.join(f'#define {name} {value}' for name,value in defs.items())+'\n'+SOURCE).encode()
    program=check(nvrtc.nvrtcCreateProgram(source,b'vector_gemv.cu',0,[],[]))
    options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
    result=nvrtc.nvrtcCompileProgram(program,len(options),options)
    size=check(nvrtc.nvrtcGetProgramLogSize(program));log=b' '*size;check(nvrtc.nvrtcGetProgramLog(program,log))
    if int(result[0]):raise RuntimeError(log.decode())
    size=check(nvrtc.nvrtcGetCUBINSize(program));blob=b' '*size;check(nvrtc.nvrtcGetCUBIN(program,blob))
    size=check(nvrtc.nvrtcGetPTXSize(program));ptx=b' '*size;check(nvrtc.nvrtcGetPTX(program,ptx));check(nvrtc.nvrtcDestroyProgram(program))
    module=check(cu.cuModuleLoadData(blob));fn=check(cu.cuModuleGetFunction(module,b'vector_gemv'))
    attr=cu.CUfunction_attribute
    metadata={name:check(cu.cuFuncGetAttribute(value,fn)) for name,value in [
        ('registers',attr.CU_FUNC_ATTRIBUTE_NUM_REGS),('shared_bytes',attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),('local_bytes',attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]}
    text=ptx.decode();metadata.update(ptx_fma_rn_f32=text.count('fma.rn.f32'),ptx_vector_loads=sum('ld.global.' in line and ('.v2.' in line or '.v4.' in line) for line in text.splitlines()),ptx_mma_instructions=text.count('mma.sync')+text.count('wgmma.')+text.count('tcgen05.mma'))
    CACHE[key]=(module,fn,metadata,blob)
    return CACHE[key]


def gemv(x,w,config,return_resources=False):
    if len(config)!=5:
        raise ValueError('Expected five launch parameters')
    lanes,vec,threads,prefetch,u=config
    if (lanes not in (8,16,32) or vec not in (1,2,4,8) or vec>lanes
            or threads not in (32,64,128,256) or prefetch not in (1,2,4)
            or u not in (1,4,16)):
        raise ValueError('Unsupported launch configuration')
    if (x.ndim!=2 or w.ndim!=2 or x.dtype!=torch.float32 or w.dtype!=torch.float32
            or not x.is_cuda or x.device!=w.device or not x.is_contiguous() or not w.is_contiguous()):
        raise ValueError('Expected contiguous FP32 operands on the same CUDA device')
    n,k=w.shape
    if (not n or not k or tuple(x.shape)!=(1,k) or k%(lanes*prefetch)
            or x.data_ptr()%min(16,vec*4) or w.data_ptr()%min(16,vec*4)):
        raise ValueError('Expected aligned complete GEMV operands and prefetch windows')
    _,fn,metadata,_=compile_kernel(n,k,tuple(config))
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    values=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,out)]
    pointers=(ctypes.c_void_p*3)(*(ctypes.addressof(v) for v in values))
    cols=threads//(lanes//vec)
    check(cu.cuLaunchKernel(fn,(n+cols-1)//cols,1,1,threads,1,1,0,
        cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return (out,metadata) if return_resources else out

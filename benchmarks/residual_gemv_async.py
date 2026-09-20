"""Research native 16-lane GEMV with shared weight tiles and residual epilogue."""
import ctypes,hashlib
import torch
from fast_moss.normalization import compiler,_check

SOURCE=r'''
__device__ __forceinline__ void copy16(float* target,const float* src){
  #if ASYNC
  unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(target));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(src):"memory");
  #else
  *reinterpret_cast<float4*>(target)=*reinterpret_cast<const float4*>(src);
  #endif
}
__device__ __forceinline__ void copy_tile(float* dst,float* xv,const float* W,const float* X,int first,int start){
  constexpr int COLS=THREADS/16;
  #pragma unroll
  for(int i=threadIdx.x;i<COLS*TILE/4;i+=THREADS){
    int col=i/(TILE/4),k=(i%(TILE/4))*4;
    copy16(dst+col*(TILE+PAD)+k,W+(first+col)*K+start+k);
  }
  #if INPUT==2
  for(int i=threadIdx.x;i<TILE/4;i+=THREADS)copy16(xv+i*4,X+start+i*4);
  #endif
  #if ASYNC
  asm volatile("cp.async.commit_group;":::"memory");
  #endif
}
extern "C" __global__ void residual_gemv_async(const float* __restrict__ X,const float* __restrict__ W,
    const float* R,const float* S,float* Y){
  constexpr int COLS=THREADS/16;
  __shared__ __align__(128) float weights[STAGES][COLS*(TILE+PAD)];
  #if INPUT==1
  __shared__ __align__(128) float vector[K];
  for(int i=threadIdx.x;i<K/4;i+=THREADS)
    reinterpret_cast<float4*>(vector)[i]=reinterpret_cast<const float4*>(X)[i];
  __syncthreads();
  #elif INPUT==2
  __shared__ __align__(128) float vector[STAGES][TILE];
  #endif
  int lane=threadIdx.x%16,col=threadIdx.x/16,first=blockIdx.x*COLS;
  float acc=0.f;
  #if STAGES==2
  #if INPUT==2
  copy_tile(weights[0],vector[0],W,X,first,0);
  #else
  copy_tile(weights[0],nullptr,W,X,first,0);
  #endif
  #endif
  #pragma unroll 1
  for(int start=0;start<K;start+=TILE){
    int tile=start/TILE,slot=tile%STAGES;
    #if STAGES==2
    if(start+TILE<K){
      #if INPUT==2
      copy_tile(weights[(tile+1)%STAGES],vector[(tile+1)%STAGES],W,X,first,start+TILE);
      #else
      copy_tile(weights[(tile+1)%STAGES],nullptr,W,X,first,start+TILE);
      #endif
    }
    #if ASYNC
    if(start+TILE<K)asm volatile("cp.async.wait_group 1;":::"memory");
    else asm volatile("cp.async.wait_group 0;":::"memory");
    #endif
    #else
    #if INPUT==2
    copy_tile(weights[0],vector[0],W,X,first,start);
    #else
    copy_tile(weights[0],nullptr,W,X,first,start);
    #endif
    #if ASYNC
    asm volatile("cp.async.wait_group 0;":::"memory");
    #endif
    #endif
    __syncthreads();
    #pragma unroll UNROLL
    for(int k=lane;k<TILE;k+=16){
      #if INPUT==0
      float x=X[start+k];
      #elif INPUT==1
      float x=vector[start+k];
      #else
      float x=vector[slot][k];
      #endif
      acc=__fmaf_rn(x,weights[slot][col*(TILE+PAD)+k],acc);
    }
    __syncthreads();
  }
  #pragma unroll
  for(int delta=8;delta;delta/=2)
    acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,delta,16));
  if(lane==0){
    float out=__fadd_rn(acc,0.f);
    Y[first+col]=__fadd_rn(R[first+col],__fmul_rn(out,S[first+col]));
  }
}
'''
CACHE={}

def shared_bytes(k,config):
    threads,tile,stages,pad,unroll,asynchronous,inputs=config
    return stages*(threads//16)*(tile+pad)*4+(k*4 if inputs==1 else stages*tile*4 if inputs==2 else 0)

def compile_kernel(k,config):
    threads,tile,stages,pad,unroll,asynchronous,inputs=config
    if (k not in (1280,5120) or threads not in (32,64,128,256) or tile not in (64,128,256,320,640)
        or stages not in (1,2) or pad not in (0,16) or unroll not in (1,4,16,32)
        or asynchronous not in (0,1) or inputs not in (0,1,2) or shared_bytes(k,config)>49152):
        raise ValueError('Unsupported residual GEMV staging configuration')
    key=(torch.cuda.current_device(),k,tuple(config))
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm staged residual GEMV before capture')
    defs=dict(N=1280,K=k,THREADS=threads,TILE=tile,STAGES=stages,PAD=pad,UNROLL=unroll,ASYNC=asynchronous,INPUT=inputs)
    source=('\n'.join(f'#define {a} {b}' for a,b in defs.items())+'\n'+SOURCE).encode()
    cu,nvrtc=compiler();program=_check(nvrtc.nvrtcCreateProgram(source,b'residual_gemv_async.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
        ptx=b' '*_check(nvrtc.nvrtcGetPTXSize(program));_check(nvrtc.nvrtcGetPTX(program,ptx))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'residual_gemv_async'));a=cu.CUfunction_attribute
    resource={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    text=ptx.decode();resource.update(local_loads=text.count('ld.local'),local_stores=text.count('st.local'),matrix_instructions=sum(text.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')),async_copies=text.count('cp.async.cg.shared.global'),cubin_sha256=hashlib.sha256(blob).hexdigest())
    CACHE[key]=(module,fn,resource,blob);return CACHE[key]

def linear(x,w,residual,scale,config,resources=False):
    if (w.ndim!=2 or tuple(w.shape) not in ((1280,1280),(1280,5120)) or x.shape!=(1,w.shape[1])
        or residual.shape!=(1,1280) or scale.shape!=(1280,) or not x.is_cuda
        or any(t.device!=x.device or t.dtype!=torch.float32 or not t.is_contiguous() for t in (x,w,residual,scale))
        or x.data_ptr()%16 or w.data_ptr()%16):raise ValueError('Expected aligned native FP32 GEMV/residual operands')
    with torch.cuda.device(x.device):
        cu,_=compiler();_,fn,res,_=compile_kernel(w.shape[1],config);out=torch.empty((1,1280),device=x.device,dtype=x.dtype)
        values=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,residual,scale,out)]
        ptrs=(ctypes.c_void_p*len(values))(*(ctypes.addressof(v) for v in values));threads=config[0]
        _check(cu.cuLaunchKernel(fn,1280//(threads//16),1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(ptrs),0))
    return (out,res) if resources else out

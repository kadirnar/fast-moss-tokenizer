"""Research shared-memory weight staging for exact native-order norm/GEMV.

Each eight-thread group retains its original cyclic FMA chain. Global-to-shared
copies move FP32 words only; optional double buffering overlaps the next tile.
"""
import ctypes
import hashlib
import torch
from fast_moss.norm_projection import header,prepare
from fast_moss.normalization import compiler,_check

SOURCE=header+prepare+r'''
__device__ __forceinline__ void copy_tile(float* dst,const float* W,int first,int start){
  constexpr int COLS=THREADS/8;
  #pragma unroll
  for(int i=threadIdx.x;i<COLS*TILE/4;i+=THREADS){
    int col=i/(TILE/4),k=(i%(TILE/4))*4;
    const float* src=W+(first+col)*1280+start+k;
    float* target=dst+col*(TILE+PAD)+k;
    #if ASYNC
    unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(target));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(src):"memory");
    #else
    *reinterpret_cast<float4*>(target)=*reinterpret_cast<const float4*>(src);
    #endif
  }
  #if ASYNC
  asm volatile("cp.async.commit_group;":::"memory");
  #endif
}
extern "C" __global__ void norm_gemv_async(const float* X,const float* W,
    const float* G,const float* B,float eps,float* Y,float* D){
  constexpr int COLS=THREADS/8;
  __shared__ __align__(128) float normalized[K];
  __shared__ __align__(128) float weights[STAGES][COLS*(TILE+PAD)];
  prepare(X,G,B,eps,normalized);
  __syncthreads();
  #if DEBUG
  if(blockIdx.x==0)for(int i=threadIdx.x;i<K;i+=THREADS)D[i]=normalized[i];
  #endif
  int lane=threadIdx.x%8,col=threadIdx.x/8,first=blockIdx.x*COLS;
  float acc=0.f;
  #if STAGES==2
  copy_tile(weights[0],W,first,0);
  #endif
  #pragma unroll 1
  for(int start=0;start<K;start+=TILE){
    int tile=start/TILE,slot=tile%STAGES;
    #if STAGES==2
    if(start+TILE<K)copy_tile(weights[(tile+1)%STAGES],W,first,start+TILE);
    #if ASYNC
    if(start+TILE<K)asm volatile("cp.async.wait_group 1;":::"memory");
    else asm volatile("cp.async.wait_group 0;":::"memory");
    #endif
    #else
    copy_tile(weights[0],W,first,start);
    #if ASYNC
    asm volatile("cp.async.wait_group 0;":::"memory");
    #endif
    #endif
    __syncthreads();
    #pragma unroll UNROLL
    for(int k=lane;k<TILE;k+=8)
      acc=__fmaf_rn(normalized[start+k],weights[slot][col*(TILE+PAD)+k],acc);
    __syncthreads();
  }
  #pragma unroll
  for(int delta=4;delta;delta/=2)
    acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,delta,8));
  if(lane==0){
    float out=__fadd_rn(acc,0.f);
    #if MODE
    out=__fmul_rn(__fmul_rn(out,.5f),__fadd_rn(erff(__fmul_rn(out,.7071067811865476f)),1.f));
    #endif
    Y[first+col]=out;
  }
}
'''
CACHE={}

def compile_kernel(n,mode,config,debug=False):
    register,threads,tile,stages,pad,unroll,asynchronous=config
    if (n not in (3840,5120) or mode not in ('none','gelu') or register not in (0,1)
        or threads not in (32,64,128,256) or (not register and threads!=128)
        or tile not in (64,128,256,320,640) or stages not in (1,2) or pad not in (0,8)
        or unroll not in (1,4,16,32) or asynchronous not in (0,1)
        or 5120+stages*(threads//8)*(tile+pad)*4+128>49152):
        raise ValueError('Unsupported shared staging configuration')
    key=(torch.cuda.current_device(),n,mode,tuple(config),debug)
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm asynchronous norm GEMV before capture')
    defs=dict(N=n,K=1280,REGISTER_NORM=register,THREADS=threads,TILE=tile,STAGES=stages,PAD=pad,UNROLL=unroll,ASYNC=asynchronous,MODE=int(mode=='gelu'),DEBUG=int(debug))
    source=('\n'.join(f'#define {k} {v}' for k,v in defs.items())+'\n'+SOURCE).encode()
    cu,nvrtc=compiler();program=_check(nvrtc.nvrtcCreateProgram(source,b'norm_gemv_async.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=true']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
        ptx=b' '*_check(nvrtc.nvrtcGetPTXSize(program));_check(nvrtc.nvrtcGetPTX(program,ptx))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'norm_gemv_async'));a=cu.CUfunction_attribute
    resource={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    text=ptx.decode();resource.update(local_loads=text.count('ld.local'),local_stores=text.count('st.local'),matrix_instructions=sum(text.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')),async_copies=text.count('cp.async.cg.shared.global'),cubin_sha256=hashlib.sha256(blob).hexdigest())
    CACHE[key]=(module,fn,resource,blob);return CACHE[key]

def linear(x,w,g,b,eps,mode,config,debug=False,resources=False):
    if (x.shape!=(1,1280) or w.shape not in ((3840,1280),(5120,1280)) or g.shape!=(1280,) or b.shape!=(1280,)
        or not x.is_cuda or any(t.device!=x.device or t.dtype!=torch.float32 or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,g,b))):
        raise ValueError('Expected aligned one-row FP32 norm/projection operands')
    with torch.cuda.device(x.device):
        cu,_=compiler();_,fn,res,_=compile_kernel(w.shape[0],mode,config,debug)
        out=torch.empty(1,w.shape[0],device=x.device,dtype=x.dtype);d=torch.empty_like(x) if debug else out
        args=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,g,b)]+[ctypes.c_float(eps)]+[ctypes.c_void_p(t.data_ptr()) for t in (out,d)]
        pointers=(ctypes.c_void_p*len(args))(*(ctypes.addressof(v) for v in args))
        threads=config[1];cols=threads//8
        _check(cu.cuLaunchKernel(fn,w.shape[0]//cols,1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    result=(out,d) if debug else out
    return (result,res) if resources else result

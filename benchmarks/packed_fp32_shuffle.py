"""Research-only CUDA cooperative fixed28 reconstruction and exact FP32 GEMV."""
import ctypes
import torch
from cuda.bindings import driver as cu,nvrtc
from benchmarks.small_vector_cuda import check

SOURCE=r'''
extern "C" __global__ void packed_gemv(const float* X,const unsigned* DATA,
                                      const unsigned long long* H,float* Y){
  int lane=threadIdx.x%LANES;
  int n=blockIdx.x*(THREADS/LANES)+threadIdx.x/LANES;
  float acc=0.f;
  for(int part=0;part<K/256;++part){
    unsigned long long h=n<N?H[n*(K/256)+part]:0;
    unsigned offset=unsigned(h>>16),base=unsigned(h&255);
    bool packed=((h>>8)&255)==28;
    #pragma unroll UNROLL
    for(int step=0;step<256/LANES;++step){
      int local=step*LANES+lane,slot=lane%8;
      unsigned word=0;
      if(n<N && (!packed || slot<7))word=DATA[offset+(packed?(local/8)*7+slot:local)];
      unsigned prev=__shfl_up_sync(0xffffffff,word,1);
      unsigned a=slot==0?word:prev;
      unsigned q=__funnelshift_r(a,word,(lane*28)%32)&0xfffffff;
      unsigned raw=(q&0x7fffff)|((q&0x800000)<<8)|(((q>>24)+base)<<23);
      float w=__uint_as_float(packed?raw:word);
      acc=__fmaf_rn(X[part*256+local],w,acc);
    }
  }
  #pragma unroll
  for(int delta=LANES/2;delta>0;delta/=2)
    acc=__fadd_rn(acc,__shfl_xor_sync(0xffffffff,acc,delta,LANES));
  acc=__fadd_rn(acc,0.f);
  if(lane==0 && n<N)Y[n]=acc;
}
'''
CACHE={}


def compile_kernel(n,k,config):
    lanes,threads,u=config;key=(n,k,config)
    if key in CACHE:return CACHE[key]
    source=('\n'.join(f'#define {name} {value}' for name,value in dict(N=n,K=k,LANES=lanes,THREADS=threads,UNROLL=u).items())+'\n'+SOURCE).encode()
    program=check(nvrtc.nvrtcCreateProgram(source,b'packed_gemv.cu',0,[],[]))
    options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
    result=nvrtc.nvrtcCompileProgram(program,len(options),options)
    size=check(nvrtc.nvrtcGetProgramLogSize(program));log=b' '*size
    check(nvrtc.nvrtcGetProgramLog(program,log))
    if int(result[0]):raise RuntimeError(log.decode())
    size=check(nvrtc.nvrtcGetCUBINSize(program));blob=b' '*size;check(nvrtc.nvrtcGetCUBIN(program,blob))
    size=check(nvrtc.nvrtcGetPTXSize(program));ptx=b' '*size;check(nvrtc.nvrtcGetPTX(program,ptx))
    check(nvrtc.nvrtcDestroyProgram(program))
    module=check(cu.cuModuleLoadData(blob));fn=check(cu.cuModuleGetFunction(module,b'packed_gemv'))
    attr=cu.CUfunction_attribute
    metadata={name:check(cu.cuFuncGetAttribute(value,fn)) for name,value in [
        ('registers',attr.CU_FUNC_ATTRIBUTE_NUM_REGS),('shared_bytes',attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        ('local_bytes',attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]}
    text=ptx.decode();metadata.update(ptx_fma_rn_f32=text.count('fma.rn.f32'),ptx_mma_instructions=text.count('mma.sync')+text.count('wgmma.')+text.count('tcgen05.mma'))
    CACHE[key]=(module,fn,metadata)
    return CACHE[key]


def gemv(x,packed,config,return_kernel=False):
    if packed.mode!='fixed28':raise ValueError('Cooperative decoder requires fixed28 blocks')
    n,k=packed.shape;lanes,threads,u=config
    if tuple(x.shape)!=(1,k) or k%256:raise ValueError('Expected one complete GEMV row')
    _,fn,metadata=compile_kernel(n,k,tuple(config))
    out=torch.empty((1,n),device=x.device,dtype=x.dtype)
    values=[ctypes.c_void_p(t.data_ptr()) for t in (x,packed.data,packed.headers,out)]
    pointers=(ctypes.c_void_p*4)(*(ctypes.addressof(v) for v in values))
    cols=threads//lanes
    check(cu.cuLaunchKernel(fn,(n+cols-1)//cols,1,1,threads,1,1,0,
        cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return (out,metadata) if return_kernel else out

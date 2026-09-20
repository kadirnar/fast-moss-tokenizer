"""Exact one-row normalization/projection fusion on native FP32 weights.

Each CTA reconstructs the native LayerNorm Welford tree and affine values in
shared memory, then runs the native eight-lane GEMV reduction. The CTA barrier
precedes every consumer; each normalized shared-memory slot has one writer.
Only the two measured configurations below are selected by runtime dispatch.
"""
import ctypes
import torch
from .normalization import SOURCE as NORM_SOURCE,compiler,_check
GEMV_SOURCE = r'''
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

# Retain the existing Welford arithmetic and normalized affine expression.
header=NORM_SOURCE[:NORM_SOURCE.index('extern "C"')]
body=NORM_SOURCE[NORM_SOURCE.index('    int lane='):NORM_SOURCE.index('    if((REGISTER&&lane==0)')]
body=body.replace('int row=REGISTER ? blockIdx.x*(THREADS/32)+warp : blockIdx.x;','int row=0;').replace('    if(row>=rows)return;\n','')
body=body.replace('REGISTER','REGISTER_NORM').replace('N/','K/').replace('float(N)','float(K)').replace('row*N','row*K')
body=body.replace('int start=REGISTER_NORM?lane:threadIdx.x,stride=REGISTER_NORM?32:128;','int start=threadIdx.x,stride=THREADS;')
body=body.replace('#pragma unroll UNROLL','#pragma unroll 1')
prepare='__device__ __forceinline__ void prepare(const float* X,const float* G,const float* B,float eps,float* Y){\n'+body+'}\n'
s=GEMV_SOURCE.replace('void vector_gemv(', 'void norm_gemv(')
s=s.replace('const float* __restrict__ W,float* Y)', 'const float* __restrict__ W,const float* G,const float* B,float eps,float* Y,float* D)')
s=s.replace('  constexpr int THREAD_LANES=LANES/VEC;', '''  __shared__ __align__(16) float normalized[K];
  prepare(X,G,B,eps,normalized);
  __syncthreads();
  #if DEBUG
  if(blockIdx.x==0)for(int i=threadIdx.x;i<K;i+=THREADS)D[i]=normalized[i];
  #endif
  constexpr int THREAD_LANES=LANES/VEC;''')
s=s.replace('fetch(X+k,xs[p]);','fetch(normalized+k,xs[p]);')
s=s.replace('if(lane==0 && n<N)Y[n]=__fadd_rn(acc[0],0.f);', '''if(lane==0 && n<N){
    float out=__fadd_rn(acc[0],0.f);
    #if MODE
    out=__fmul_rn(__fmul_rn(out,.5f),__fadd_rn(erff(__fmul_rn(out,.7071067811865476f)),1.f));
    #endif
    Y[n]=out;
  }''')
SOURCE=header+prepare+s
CONFIGS={'none':(0,1,128,2,4),'gelu':(1,1,256,2,4)}
CACHE={}


def compile_kernel(n,mode,config,debug=False):
    register,vec,threads,prefetch,unroll=config
    if (register not in (0,1) or vec not in (1,2,4,8) or threads not in (32,64,128,256)
        or (not register and threads!=128) or prefetch not in (1,2,4) or unroll not in (1,4,16,32)):
        raise ValueError('Unsupported fused norm GEMV configuration')
    key=(torch.cuda.current_device(),n,mode,tuple(config),debug)
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm fused norm GEMV before capture')
    defs=dict(N=n,K=1280,LANES=8,VEC=vec,THREADS=threads,PREFETCH=prefetch,UNROLL=unroll,REGISTER_NORM=register,MODE=int(mode=='gelu'),DEBUG=int(debug))
    source=('\n'.join(f'#define {k} {v}' for k,v in defs.items())+'\n'+SOURCE).encode()
    cu,nvrtc=compiler();program=_check(nvrtc.nvrtcCreateProgram(source,b'norm_gemv.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=true']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
        ptx=b' '*_check(nvrtc.nvrtcGetPTXSize(program));_check(nvrtc.nvrtcGetPTX(program,ptx))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'norm_gemv'));a=cu.CUfunction_attribute
    resource={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    text=ptx.decode();resource.update(local_loads=text.count('ld.local'),local_stores=text.count('st.local'),matrix_instructions=sum(text.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')))
    CACHE[key]=(module,fn,resource,blob);return CACHE[key]


def linear(x,w,g,b,eps,mode,config=None,debug=False,resources=False):
    config=CONFIGS.get(mode) if config is None else config
    if (x.shape!=(1,1280) or w.shape not in ((3840,1280),(5120,1280)) or g.shape!=(1280,) or b.shape!=(1280,)
        or mode not in ('none','gelu') or not x.is_cuda or any(t.device!=x.device or t.dtype!=torch.float32 or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,g,b))):
        raise ValueError('Expected aligned one-row FP32 norm/projection operands')
    with torch.cuda.device(x.device):
        cu,_=compiler();_,fn,res,_=compile_kernel(w.shape[0],mode,config,debug)
        out=torch.empty(1,w.shape[0],device=x.device,dtype=x.dtype);d=torch.empty_like(x) if debug else out
        args=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,g,b)]+[ctypes.c_float(eps)]+[ctypes.c_void_p(t.data_ptr()) for t in (out,d)]
        pointers=(ctypes.c_void_p*len(args))(*(ctypes.addressof(v) for v in args))
        _,vec,threads,_,_=config;cols=threads//(8//vec)
        _check(cu.cuLaunchKernel(fn,(w.shape[0]+cols-1)//cols,1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    result=(out,d) if debug else out
    return (result,res) if resources else result


def project(runtime,x,norm,module,mode):
    """Return a fused projection only through active, unmodified runtime owners."""
    from threading import get_ident
    from .normalization import owned_forward
    from torch.nn.modules import module as hooks
    if not runtime.active or get_ident()!=runtime.thread:
        raise RuntimeError('Use normalization/projection on the active matrix owner thread')
    if (runtime.backend!='cuda' or not runtime.norm_gemv_enabled
        or tuple(x.shape)!=(1,1,1280) or x.device!=runtime.device or x.dtype!=torch.float32
        or not x.is_contiguous() or x.data_ptr()%16 or x.requires_grad or torch.is_autocast_enabled('cuda')
        or type(norm) is not torch.nn.LayerNorm or tuple(norm.normalized_shape)!=(1280,)
        or not owned_forward(norm) or norm.eps!=1e-5
        or type(module) is not torch.nn.Linear or runtime.forwards.get(module) is not module.forward
        or module.weight.device!=x.device or module.weight.dtype!=x.dtype
        or module.bias is not None or tuple(module.weight.shape)!=((3840,1280) if mode=='none' else (5120,1280))
        or not module.weight.is_contiguous() or module.weight.data_ptr()%256
        or hooks._global_forward_hooks or hooks._global_forward_pre_hooks
        or any(m._forward_hooks or m._forward_pre_hooks for m in (norm,module))):return None
    g,b=norm.weight,norm.bias
    if (g is None or b is None or any(t.shape!=(1280,) or t.device!=x.device or t.dtype!=x.dtype
        or not t.is_contiguous() or t.data_ptr()%16 for t in (g,b))
        or any(t.requires_grad for t in (g,b,module.weight))):return None
    stream=torch.cuda.current_stream(x.device).cuda_stream
    key=(id(module),mode,stream)
    if torch.cuda.is_current_stream_capturing() and key not in runtime.norm_gemv_warmed:
        raise RuntimeError('Warm normalization/projection on the capture stream')
    out=linear(x.reshape(1,1280),module.weight,g,b,norm.eps,mode)
    runtime.norm_gemv_warmed.add(key);runtime.norm_gemv_calls+=1
    if mode=='none':runtime.norm_qkv_calls+=1
    else:runtime.norm_ffn_calls+=1
    return out.reshape(1,1,module.out_features)


def attention_block(self,x):
    from torch.nn.modules import module as hooks
    from .attention import forward as attention_forward
    projection=self.self_attn.in_projs[0]
    if (x.requires_grad or torch.is_autocast_enabled('cuda')
        or hooks._global_forward_hooks or hooks._global_forward_pre_hooks
        or any(m._forward_hooks or m._forward_pre_hooks for m in (self.norm1,self.self_attn,projection,self.layer_scale_1))):
        return self._fast_observed_sa(x)
    if getattr(self.self_attn.forward,'__func__',None) is not attention_forward:
        return self._fast_original_sa(x)
    projected=project(self._fast_ffn_runtime,x,self.norm1,projection,'none')
    if projected is None:return self._fast_original_sa(x)
    update=self.self_attn(x,x,x,_fast_projected=projected)
    return self._fast_scale_add(x,update,self.layer_scale_1.scale)

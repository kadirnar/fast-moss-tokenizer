"""Research cache-policy variants of exact norm/projection fusion."""
import ctypes
import torch
from fast_moss.normalization import SOURCE as NORM_SOURCE,compiler,_check
from benchmarks.gemv_vector import SOURCE as GEMV_SOURCE

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

WEIGHT_FETCH = r'''
__device__ __forceinline__ void fetch_weight(const float* p,float (&v)[VEC]){
  #if VEC==1
  asm volatile("ld.global." CACHEOP ".f32 %0, [%1];":"=f"(v[0]):"l"(p));
  #elif VEC==2
  asm volatile("ld.global." CACHEOP ".v2.f32 {%0,%1}, [%2];":"=f"(v[0]),"=f"(v[1]):"l"(p));
  #else
  #pragma unroll
  for(int i=0;i<VEC;i+=4)
    asm volatile("ld.global." CACHEOP ".v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v[i]),"=f"(v[i+1]),"=f"(v[i+2]),"=f"(v[i+3]):"l"(p+i));
  #endif
}
'''
SOURCE=WEIGHT_FETCH+SOURCE.replace('fetch(W+n*K+k,ws[p]);','fetch_weight(W+n*K+k,ws[p]);')

CACHE={}


def compile_kernel(n,mode,config,debug=False):
    register,vec,threads,prefetch,unroll,cache=config
    if (cache not in ('ca','cg','cs') or register not in (0,1) or vec not in (1,2,4,8) or threads not in (32,64,128,256)
        or (not register and threads!=128) or prefetch not in (1,2,4) or unroll not in (1,4,16,32)):
        raise ValueError('Unsupported fused norm GEMV configuration')
    key=(torch.cuda.current_device(),n,mode,tuple(config),debug)
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm fused norm GEMV before capture')
    defs=dict(CACHEOP='"'+cache+'"',N=n,K=1280,LANES=8,VEC=vec,THREADS=threads,PREFETCH=prefetch,UNROLL=unroll,REGISTER_NORM=register,MODE=int(mode=='gelu'),DEBUG=int(debug))
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
    text=ptx.decode();resource['cache_load_instructions']=sum(('ld.global.'+cache) in line for line in text.splitlines());resource.update(local_loads=text.count('ld.local'),local_stores=text.count('st.local'),matrix_instructions=sum(text.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')))
    CACHE[key]=(module,fn,resource,blob);return CACHE[key]


def linear(x,w,g,b,eps,mode,config,debug=False,resources=False):
    if (x.shape!=(1,1280) or w.shape not in ((3840,1280),(5120,1280)) or g.shape!=(1280,) or b.shape!=(1280,)
        or mode not in ('none','gelu') or not x.is_cuda or any(t.device!=x.device or t.dtype!=torch.float32 or not t.is_contiguous() or t.data_ptr()%16 for t in (x,w,g,b))):
        raise ValueError('Expected aligned one-row FP32 norm/projection operands')
    with torch.cuda.device(x.device):
        cu,_=compiler();_,fn,res,_=compile_kernel(w.shape[0],mode,config,debug)
        out=torch.empty(1,w.shape[0],device=x.device,dtype=x.dtype);d=torch.empty_like(x) if debug else out
        args=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,g,b)]+[ctypes.c_float(eps)]+[ctypes.c_void_p(t.data_ptr()) for t in (out,d)]
        pointers=(ctypes.c_void_p*len(args))(*(ctypes.addressof(v) for v in args))
        _,vec,threads,_,_,_=config;cols=threads//(8//vec)
        _check(cu.cuLaunchKernel(fn,(w.shape[0]+cols-1)//cols,1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    result=(out,d) if debug else out
    return (result,res) if resources else result


def current_norm(x,g,b,eps):
    """Same kernel, output allocation and statistics allocation as runtime LayerNorm."""
    from fast_moss.normalization import _compile,CONFIGS
    cu,_=compiler();_,fn=_compile(x.device,1280,CONFIGS[(1,1280)],compiler())
    y=torch.empty_like(x);mean=torch.empty(1,device=x.device,dtype=x.dtype);rs=torch.empty_like(mean)
    values=[ctypes.c_void_p(t.data_ptr()) for t in (x,g,b,y,mean,rs)]+[ctypes.c_float(eps),ctypes.c_int(1)]
    pointers=(ctypes.c_void_p*len(values))(*(ctypes.addressof(v) for v in values))
    _check(cu.cuLaunchKernel(fn,1,1,1,128,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return y

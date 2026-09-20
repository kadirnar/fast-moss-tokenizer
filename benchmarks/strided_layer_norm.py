"""Research strided loads/shared transpose around the exact native Welford tree."""
import ctypes
import torch
from fast_moss.normalization import SOURCE as BASE, compiler, _check

SOURCE=BASE.replace('float4 v=reinterpret_cast<const float4*>(x)[logical+step*128];',
                    'float4 v=read4(x,logical+step*128);')
SOURCE=SOURCE.replace('X+row*N,logical','row_ptr,logical')
SOURCE=SOURCE.replace('float4 v=reinterpret_cast<const float4*>(X+row*N)[idx];','float4 v=read4(row_ptr,idx);')
SOURCE=SOURCE.replace('    if(row>=rows)return;',r'''
    #if STAGE
    constexpr int R=THREADS/32;
    __shared__ float tile[R*(N+PAD)];
    for(int index=threadIdx.x;index<N*R;index+=THREADS){
        int c=index/R,r=index%R,global_row=blockIdx.x*R+r;
        tile[r*(N+PAD)+c]=global_row<rows ? X[(global_row/T)*N*T+(global_row%T)+c*T] : 0.f;
    }
    __syncthreads();
    const float* row_ptr=tile+warp*(N+PAD);
    #else
    const float* row_ptr=X+(row/T)*N*T+(row%T);
    #endif
    if(row>=rows)return;''')
SOURCE=r'''
__device__ __forceinline__ float4 read4(const float* x,int vector){
    #if STAGE
    return reinterpret_cast<const float4*>(x)[vector];
    #else
    int c=vector*4;
    return {x[c*T],x[(c+1)*T],x[(c+2)*T],x[(c+3)*T]};
    #endif
}
'''+SOURCE
CACHE={}


def compile_kernel(n,t,config):
    device=torch.cuda.current_device();key=(device,n,t,config)
    if key in CACHE:return CACHE[key]
    register,threads,unroll,stage,pad=config;cu,nvrtc=compiler()
    defs=dict(N=n,T=t,REGISTER=int(register),THREADS=threads,UNROLL=unroll,STAGE=int(stage),PAD=pad)
    source=('\n'.join(f'#define {k} {v}' for k,v in defs.items())+'\n'+SOURCE).encode()
    program=_check(nvrtc.nvrtcCreateProgram(source,b'strided_layer_norm.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=true']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'layer_norm'))
    a=cu.CUfunction_attribute
    resources={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [
        ('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    CACHE[key]=(module,fn,resources);return CACHE[key]


def layer_norm(x,g,b,eps=1e-5,config=(False,128,1,False,0)):
    register,threads,unroll,stage,pad=config
    if (x.ndim!=3 or x.shape[-1] not in (768,1280) or x.shape[1]<2
            or x.stride()!=(x.shape[1]*x.shape[2],1,x.shape[1]) or not x.is_cuda or x.dtype!=torch.float32
            or any(v.dtype!=x.dtype or v.device!=x.device or not v.is_contiguous() or v.shape!=(x.shape[-1],) or v.data_ptr()%16 for v in (g,b))
            or (stage and not register) or (not register and threads!=128)
            or threads not in (32,64,128,256) or pad not in (0,4) or unroll not in (1,4)):
        raise ValueError('Expected a dense transposed BTC FP32 LayerNorm input')
    _,t,n=x.shape;rows=x.shape[0]*t
    with torch.cuda.device(x.device):
        cu,_=compiler();_,fn,_=compile_kernel(n,t,config)
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype);mean=torch.empty(rows,device=x.device);rs=torch.empty_like(mean)
        vals=[ctypes.c_void_p(v.data_ptr()) for v in (x,g,b,y,mean,rs)]+[ctypes.c_float(eps),ctypes.c_int(rows)]
        pointers=(ctypes.c_void_p*len(vals))(*(ctypes.addressof(v) for v in vals))
        grid=(rows+threads//32-1)//(threads//32) if register else rows
        _check(cu.cuLaunchKernel(fn,grid,1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return y,mean,rs

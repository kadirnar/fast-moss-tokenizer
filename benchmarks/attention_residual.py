"""Research attention output projections with exact scale/residual epilogues.

Reuses native reduction schedules, including dense transposed BTC outputs.
No production dispatch/configuration is changed by importing this module.
"""
import ctypes,hashlib
import torch
import triton
from fast_moss.small_matrices import CONFIGS as SMALL,linear as small
from fast_moss.cuda_matrices import CONFIGS as CUDA,linear as cuda
from fast_moss.ffn import _ffn_gemv
from fast_moss.strided_ffn import _strided_ffn_fixed,cuda_source
from fast_moss.normalization import compiler,_check
from fast_moss.kernels import scale_add

CONFIGS={shape:(v[0],v[1:]) for shape,v in SMALL.items() if shape[1]==shape[2]}
CONFIGS.update({shape:('cuda',v) for shape,v in CUDA.items() if shape[1]==shape[2]})
CACHE={}
SOURCE=cuda_source('cuda').replace('strided_ffn_cuda','attention_residual')

def compile_kernel(shape,time):
    key=(torch.cuda.current_device(),shape,time)
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm attention residual before capture')
    m,n,k=shape;rows,cols,u,distribute=CONFIGS[shape][1]
    defs=dict(T=time,M=m,N=n,K=k,ROWS=rows,COLS=cols,GROUPS=k//256,UNROLL=u,DISTRIBUTE=int(distribute),MODE=1)
    source=('\n'.join(f'#define {a} {b}' for a,b in defs.items())+'\n'+SOURCE).encode()
    cu,nvrtc=compiler();program=_check(nvrtc.nvrtcCreateProgram(source,b'attention_residual.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
        ptx=b' '*_check(nvrtc.nvrtcGetPTXSize(program));_check(nvrtc.nvrtcGetPTX(program,ptx))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob));fn=_check(cu.cuModuleGetFunction(module,b'attention_residual'));a=cu.CUfunction_attribute
    info={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    text=ptx.decode();info.update(local_loads=text.count('ld.local'),local_stores=text.count('st.local'),matrix_instructions=sum(text.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')),cubin_sha256=hashlib.sha256(blob).hexdigest())
    CACHE[key]=(module,fn,info);return CACHE[key]

def current(x,w,residual,scale):
    shape=(x.shape[0],*w.shape)
    y=cuda(x,w,compiler()) if shape in CUDA else small(x,w)
    return scale_add(residual,y.reshape(residual.shape),scale)

def linear(x,w,residual,scale,resources=False):
    if x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1]:raise ValueError('Expected matrix operands')
    m,k=x.shape;n=w.shape[0];shape=(m,n,k)
    if (shape not in CONFIGS or not x.is_cuda or any(t.dtype!=torch.float32 or t.device!=x.device for t in (x,w,residual,scale))
        or not all(t.is_contiguous() for t in (x,w,scale)) or residual.ndim!=3 or residual.shape[-1]!=n or residual.numel()!=m*n or scale.shape!=(n,)):
        raise ValueError('Unsupported FP32 attention residual operands')
    time=1 if residual.is_contiguous() else residual.shape[1]
    if not residual.is_contiguous() and residual.stride()!=(time*n,1,time):raise ValueError('Unsupported residual strides')
    family,cfg=CONFIGS[shape]
    with torch.cuda.device(x.device):
        # Native pointwise arithmetic canonicalizes contiguous singleton strides.
        out=(torch.empty(residual.shape,device=x.device,dtype=x.dtype) if residual.is_contiguous()
             else torch.empty_strided(residual.shape,residual.stride(),device=x.device,dtype=x.dtype))
        if family=='gemv':
            lanes,cols,warps,u=cfg
            kernel=_ffn_gemv[(triton.cdiv(n,cols),)](x,w,residual,scale,out,n,k,lanes,cols,u,'residual',num_warps=warps,enable_fp_fusion=False)
        elif family=='fixed':
            rows,cols,warps,u=cfg
            kernel=_strided_ffn_fixed[(triton.cdiv(n,cols),triton.cdiv(m,rows))](x,w,residual,scale,out,m,n,k,rows,cols,u,'residual',time,num_warps=warps,enable_fp_fusion=False)
        else:
            cu,_=compiler();_,fn,info=compile_kernel(shape,time);rows,cols,_,_=cfg
            values=[ctypes.c_void_p(v.data_ptr()) for v in (x,w,residual,scale,out)];ptrs=(ctypes.c_void_p*5)(*(ctypes.addressof(v) for v in values))
            _check(cu.cuLaunchKernel(fn,triton.cdiv(n,cols),triton.cdiv(m,rows),1,16*cols*(k//256),1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(ptrs),0))
        if family!='cuda' and resources:
            ptx=kernel.asm['ptx'];info={'registers':kernel.n_regs,'local_bytes':kernel.n_spills*4,'shared_bytes':kernel.metadata.shared,'spills':kernel.n_spills,'local_loads':ptx.count('ld.local'),'local_stores':ptx.count('st.local'),'matrix_instructions':sum(ptx.count(k) for k in ('mma.sync','wgmma.','tcgen05.mma')),'cubin_sha256':hashlib.sha256(kernel.asm['cubin']).hexdigest()}
    return (out,info) if resources else out

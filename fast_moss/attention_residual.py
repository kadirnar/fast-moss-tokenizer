"""Exact attention output projections with exact scale/residual epilogues.

Reuses native reduction schedules, including dense transposed BTC outputs.
Runtime dispatch is owned by MatrixRuntime and the optimized attention block.
"""
import ctypes,hashlib
import torch
import triton
from .ffn import _ffn_gemv
from .strided_ffn import _strided_ffn_fixed,cuda_source
from .normalization import compiler,_check

# Freeze validated epilogue schedules independently of future matrix tuning.
CONFIGS = {
    (1, 1280, 1280): ('gemv', (16, 2, 1, 4)),
    (2, 768, 768): ('fixed', (2, 2, 1, 4)),
    (3, 1280, 1280): ('fixed', (3, 4, 1, 16)),
    (4, 768, 768): ('cuda', (4, 4, 16, True)),
    (6, 768, 768): ('fixed', (3, 4, 1, 16)),
    (8, 768, 768): ('fixed', (4, 4, 1, 4)),
    (8, 1280, 1280): ('fixed', (3, 4, 1, 16)),
    (12, 768, 768): ('cuda', (6, 8, 16, False)),
    (16, 768, 768): ('fixed', (4, 4, 1, 4)),
}
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



def unchanged_children(attention,runtime):
    from .rope import forward as rope_forward
    from .kv_cache import attention_complete
    for module in attention.modules():
        if module is attention:continue
        original=getattr(module,'_fast_original_rope',None)
        if module in runtime.forwards:
            if module.forward is not runtime.forwards[module]:return False
        elif original is not None:
            if (getattr(module.forward,'__func__',None) is not rope_forward
                or getattr(original,'__func__',None) is not type(module).forward):return False
        elif getattr(module.forward,'__func__',None) is not type(module).forward:return False
    original=getattr(attention,'_fast_original_complete',None)
    if original is not None:
        return (getattr(attention._complete_kv,'__func__',None) is attention_complete
                and getattr(original,'__func__',None) is type(attention)._complete_kv)
    return getattr(attention._complete_kv,'__func__',None) is type(attention)._complete_kv


def block(self,x):
    """Prepare an owned attention epilogue while preserving norm/QKV fusion."""
    from threading import get_ident
    from torch.nn.modules import module as hooks
    from .attention import forward as attention_forward
    from .normalization import owned_forward
    from .norm_projection import project
    runtime=self._fast_attention_runtime
    if not runtime.active or get_ident()!=runtime.thread:
        raise RuntimeError('Use attention residual on the active matrix owner thread')
    attention=self.self_attn
    if len(attention.in_projs)!=1 or len(attention.out_projs)!=1:
        return self._fast_attention_observed(x)
    inp,out=attention.in_projs[0],attention.out_projs[0]
    norm,scale=self.norm1,self.layer_scale_1
    watched=(norm,scale,*attention.modules())
    if (getattr(self._fast_attention_observed,'__func__',None) is not type(self)._sa_block
        or x.requires_grad or torch.is_autocast_enabled('cuda')
        or hooks._global_forward_hooks or hooks._global_forward_pre_hooks
        or any(m._forward_hooks or m._forward_pre_hooks for m in watched)
        or getattr(attention.forward,'__func__',None) is not attention_forward
        or getattr(attention._fast_original_attention,'__func__',None) is not type(attention).forward
        or runtime.forwards.get(inp) is not inp.forward or runtime.forwards.get(out) is not out.forward
        or not unchanged_children(attention,runtime)
        or type(norm) is not torch.nn.LayerNorm
        or (getattr(norm.forward,'__func__',None) is not torch.nn.LayerNorm.forward and not owned_forward(norm))
        or type(scale).__name__!='MossAudioTokenizerLayerScale'
        or getattr(scale.forward,'__func__',None) is not type(scale).forward or not getattr(scale,'channel_last',False)
        or any(p.requires_grad for m in (norm,attention,scale) for p in m.parameters())):
        return self._fast_attention_observed(x)
    if (not runtime.attention_residual_enabled or x.ndim!=3 or x.dtype!=torch.float32
        or x.device!=runtime.device or x.shape[-1] not in (768,1280)
        or (x.numel()//x.shape[-1],*out.weight.shape) not in CONFIGS
        or (not x.is_contiguous() and x.stride()!=(x.shape[1]*x.shape[2],1,x.shape[1]))
        or not out.weight.is_contiguous() or out.weight.data_ptr()%256
        or scale.scale.shape!=(x.shape[-1],) or not scale.scale.is_contiguous()
        or scale.scale.dtype!=x.dtype or scale.scale.device!=x.device):
        return self._fast_attention_previous(x)
    projected=project(runtime,x,norm,inp,'none')
    query=x if projected is not None else norm(x)
    return attention(query,query,query,_fast_projected=projected,
        _fast_epilogue=('attention_residual',x,scale.scale,self._fast_scale_add))

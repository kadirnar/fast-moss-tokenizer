"""Research exact short-row FFN epilogues on existing native matrix schedules."""
import ctypes
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from fast_moss.ffn import _mul,_add,math_library
from fast_moss.normalization import compiler,_check
from fast_moss.small_matrices import CONFIGS as SMALL
from fast_moss.cuda_matrices import CONFIGS as CUDA,SOURCE as CUDA_SOURCE
from fast_moss.wide_matrices import CONFIGS as WIDE,SOURCE as WIDE_SOURCE

PAIRS={(m,768) for m in (2,4,6,8,12,16)} | {(3,1280),(8,1280)}
CONFIGS={}
for m,width in sorted(PAIRS):
    for n,k in ((4*width,width),(width,4*width)):
        shape=(m,n,k)
        if shape in WIDE:CONFIGS[shape]=('wide',WIDE[shape])
        elif shape in CUDA:CONFIGS[shape]=('cuda',CUDA[shape])
        else:
            assert SMALL[shape][0]=='fixed'
            CONFIGS[shape]=('fixed',SMALL[shape][1:])

@triton.jit
def _epilogue(value,R,S,m,n,N:tl.constexpr,MODE:tl.constexpr,valid):
    if MODE=='gelu':
        return _mul(_mul(value,.5),_add(libdevice.erf(_mul(value,.7071067811865476)),1.))
    return _add(tl.load(R+m*N+n,valid,0),_mul(value,tl.load(S+n,valid,0)))


@triton.jit
def _short_fixed(X,W,R,S,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           ROWS:tl.constexpr,BN:tl.constexpr,UNROLL:tl.constexpr,MODE:tl.constexpr):
    base=tl.program_id(1)*ROWS
    n=tl.program_id(0)*BN+tl.arange(0,BN)
    lane=tl.arange(0,16)
    if ROWS>0:
        total0=tl.full((BN,16),0,tl.float32)
    if ROWS>1:
        total1=tl.full((BN,16),0,tl.float32)
    if ROWS>2:
        total2=tl.full((BN,16),0,tl.float32)
    if ROWS>3:
        total3=tl.full((BN,16),0,tl.float32)
    if ROWS>4:
        total4=tl.full((BN,16),0,tl.float32)
    if ROWS>5:
        total5=tl.full((BN,16),0,tl.float32)
    if ROWS>6:
        total6=tl.full((BN,16),0,tl.float32)
    if ROWS>7:
        total7=tl.full((BN,16),0,tl.float32)
    for start in range(tl.cdiv(K,256)):
        if ROWS>0:
            acc0=tl.full((BN,16),0,tl.float32)
        if ROWS>1:
            acc1=tl.full((BN,16),0,tl.float32)
        if ROWS>2:
            acc2=tl.full((BN,16),0,tl.float32)
        if ROWS>3:
            acc3=tl.full((BN,16),0,tl.float32)
        if ROWS>4:
            acc4=tl.full((BN,16),0,tl.float32)
        if ROWS>5:
            acc5=tl.full((BN,16),0,tl.float32)
        if ROWS>6:
            acc6=tl.full((BN,16),0,tl.float32)
        if ROWS>7:
            acc7=tl.full((BN,16),0,tl.float32)
        for step in tl.range(16,loop_unroll_factor=UNROLL):
            k=start*256+step*16+lane
            w=tl.load(W+n[:,None]*K+k[None,:],(n[:,None]<N)&(k[None,:]<K),0)
            if ROWS>0:
                x0=tl.load(X+(base+0)*K+k,(base+0<M)&(k<K),0)
                acc0=tl.fma(x0[None,:],w,acc0)
            if ROWS>1:
                x1=tl.load(X+(base+1)*K+k,(base+1<M)&(k<K),0)
                acc1=tl.fma(x1[None,:],w,acc1)
            if ROWS>2:
                x2=tl.load(X+(base+2)*K+k,(base+2<M)&(k<K),0)
                acc2=tl.fma(x2[None,:],w,acc2)
            if ROWS>3:
                x3=tl.load(X+(base+3)*K+k,(base+3<M)&(k<K),0)
                acc3=tl.fma(x3[None,:],w,acc3)
            if ROWS>4:
                x4=tl.load(X+(base+4)*K+k,(base+4<M)&(k<K),0)
                acc4=tl.fma(x4[None,:],w,acc4)
            if ROWS>5:
                x5=tl.load(X+(base+5)*K+k,(base+5<M)&(k<K),0)
                acc5=tl.fma(x5[None,:],w,acc5)
            if ROWS>6:
                x6=tl.load(X+(base+6)*K+k,(base+6<M)&(k<K),0)
                acc6=tl.fma(x6[None,:],w,acc6)
            if ROWS>7:
                x7=tl.load(X+(base+7)*K+k,(base+7<M)&(k<K),0)
                acc7=tl.fma(x7[None,:],w,acc7)
        if ROWS>0:
            total0=total0+acc0
        if ROWS>1:
            total1=total1+acc1
        if ROWS>2:
            total2=total2+acc2
        if ROWS>3:
            total3=total3+acc3
        if ROWS>4:
            total4=total4+acc4
        if ROWS>5:
            total5=total5+acc5
        if ROWS>6:
            total6=total6+acc6
        if ROWS>7:
            total7=total7+acc7
    if ROWS>0:
        out0=tl.reshape(tl.gather(total0,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out0=out0+tl.reshape(tl.gather(total0,tl.full((BN,1),j,tl.int32),1),(BN,))
        out0=_epilogue(out0,R,S,base+0,n,N,MODE,(base+0<M)&(n<N))
        tl.store(Y+(base+0)*N+n,out0,(base+0<M)&(n<N))
    if ROWS>1:
        out1=tl.reshape(tl.gather(total1,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out1=out1+tl.reshape(tl.gather(total1,tl.full((BN,1),j,tl.int32),1),(BN,))
        out1=_epilogue(out1,R,S,base+1,n,N,MODE,(base+1<M)&(n<N))
        tl.store(Y+(base+1)*N+n,out1,(base+1<M)&(n<N))
    if ROWS>2:
        out2=tl.reshape(tl.gather(total2,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out2=out2+tl.reshape(tl.gather(total2,tl.full((BN,1),j,tl.int32),1),(BN,))
        out2=_epilogue(out2,R,S,base+2,n,N,MODE,(base+2<M)&(n<N))
        tl.store(Y+(base+2)*N+n,out2,(base+2<M)&(n<N))
    if ROWS>3:
        out3=tl.reshape(tl.gather(total3,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out3=out3+tl.reshape(tl.gather(total3,tl.full((BN,1),j,tl.int32),1),(BN,))
        out3=_epilogue(out3,R,S,base+3,n,N,MODE,(base+3<M)&(n<N))
        tl.store(Y+(base+3)*N+n,out3,(base+3<M)&(n<N))
    if ROWS>4:
        out4=tl.reshape(tl.gather(total4,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out4=out4+tl.reshape(tl.gather(total4,tl.full((BN,1),j,tl.int32),1),(BN,))
        out4=_epilogue(out4,R,S,base+4,n,N,MODE,(base+4<M)&(n<N))
        tl.store(Y+(base+4)*N+n,out4,(base+4<M)&(n<N))
    if ROWS>5:
        out5=tl.reshape(tl.gather(total5,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out5=out5+tl.reshape(tl.gather(total5,tl.full((BN,1),j,tl.int32),1),(BN,))
        out5=_epilogue(out5,R,S,base+5,n,N,MODE,(base+5<M)&(n<N))
        tl.store(Y+(base+5)*N+n,out5,(base+5<M)&(n<N))
    if ROWS>6:
        out6=tl.reshape(tl.gather(total6,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out6=out6+tl.reshape(tl.gather(total6,tl.full((BN,1),j,tl.int32),1),(BN,))
        out6=_epilogue(out6,R,S,base+6,n,N,MODE,(base+6<M)&(n<N))
        tl.store(Y+(base+6)*N+n,out6,(base+6<M)&(n<N))
    if ROWS>7:
        out7=tl.reshape(tl.gather(total7,tl.full((BN,1),0,tl.int32),1),(BN,))
        for j in tl.static_range(1,16):
            out7=out7+tl.reshape(tl.gather(total7,tl.full((BN,1),j,tl.int32),1),(BN,))
        out7=_epilogue(out7,R,S,base+7,n,N,MODE,(base+7<M)&(n<N))
        tl.store(Y+(base+7)*N+n,out7,(base+7<M)&(n<N))

HELPERS=r'''
__device__ __forceinline__ float mul(float a,float b){
    float y;asm("mul.rn.f32 %0, %1, %2;":"=f"(y):"f"(a),"f"(b));return y;
}
__device__ __forceinline__ float epilogue(float v,const float* R,const float* S,int m,int n){
#if MODE==0
    return mul(mul(v,.5f),add(erff(mul(v,.7071067811865476f)),1.f));
#else
    return add(R[m*N+n],mul(v,S[n]));
#endif
}
'''

def cuda_source(family):
    s=WIDE_SOURCE if family=='wide' else CUDA_SOURCE
    signature='const float* X,const float* W,float* Y'
    store='Y[(m+r)*N+n]=value;'
    assert s.count(signature)==s.count(store)==1
    s=s.replace(signature,'const float* X,const float* W,const float* R,const float* S,float* Y')
    s=s.replace(store,'Y[(m+r)*N+n]=epilogue(value,R,S,m+r,n);')
    s=s.replace('void wide_cta(', 'void short_ffn_cuda(').replace('void cta_tiled(', 'void short_ffn_cuda(')
    pos=s.index('extern "C"');return s[:pos]+HELPERS+s[pos:]

CACHE={}

def compile_kernel(shape,mode,bindings):
    key=(torch.cuda.current_device(),shape,mode,CONFIGS[shape])
    if key in CACHE:return CACHE[key]
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm short FFN matrices before capture')
    family,config=CONFIGS[shape];m,n,k=shape
    if family=='wide':rows,cols,groups,unroll,distribute=config
    else:
        rows,cols,unroll,distribute=config;groups=k//256
    definitions=dict(M=m,N=n,K=k,ROWS=rows,COLS=cols,GROUPS=groups,UNROLL=unroll,DISTRIBUTE=int(distribute),MODE=int(mode=='residual'))
    source=('\n'.join(f'#define {a} {b}' for a,b in definitions.items())+'\n'+cuda_source(family)).encode()
    cu,nvrtc=bindings;program=_check(nvrtc.nvrtcCreateProgram(source,b'short_ffn.cu',0,[],[]))
    try:
        options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
        result=nvrtc.nvrtcCompileProgram(program,len(options),options)
        if int(result[0]):
            log=b' '*_check(nvrtc.nvrtcGetProgramLogSize(program));_check(nvrtc.nvrtcGetProgramLog(program,log));raise RuntimeError(log.decode())
        blob=b' '*_check(nvrtc.nvrtcGetCUBINSize(program));_check(nvrtc.nvrtcGetCUBIN(program,blob))
    finally:_check(nvrtc.nvrtcDestroyProgram(program))
    module=_check(cu.cuModuleLoadData(blob))
    try:fn=_check(cu.cuModuleGetFunction(module,b'short_ffn_cuda'))
    except BaseException:
        _check(cu.cuModuleUnload(module));raise
    a=cu.CUfunction_attribute
    resources={name:_check(cu.cuFuncGetAttribute(attr,fn)) for name,attr in [
        ('registers',a.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',a.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',a.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    CACHE[key]=(module,fn,resources);return CACHE[key]


def linear(x,w,mode,residual=None,scale=None,library=None,bindings=None,resources=False):
    if (x.ndim!=2 or w.ndim!=2 or x.shape[1]!=w.shape[1]
        or not x.is_cuda or x.dtype!=torch.float32 or w.dtype!=x.dtype or w.device!=x.device
        or not x.is_contiguous() or not w.is_contiguous()):raise ValueError('Expected supported contiguous FP32 short FFN operands')
    m,k=x.shape;n=w.shape[0];shape=(m,n,k)
    if shape not in CONFIGS or mode!=('gelu' if n>k else 'residual'):raise ValueError('Expected supported short FFN geometry and epilogue')
    if mode=='residual' and (residual is None or scale is None or residual.shape!=(m,n) or scale.shape!=(n,)
        or any(t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous() for t in (residual,scale))):raise ValueError('Expected matching residual and scale')
    family,config=CONFIGS[shape]
    with torch.cuda.device(x.device):
        out=torch.empty((m,n),device=x.device,dtype=x.dtype);r=x if residual is None else residual;s=x if scale is None else scale
        if family=='fixed':
            rows,cols,warps,unroll=config
            kernel=_short_fixed[(triton.cdiv(n,cols),triton.cdiv(m,rows))](x,w,r,s,out,m,n,k,rows,cols,unroll,mode,
                num_warps=warps,enable_fp_fusion=False,extern_libs={'libdevice':library or math_library()})
            info={'registers':kernel.n_regs,'local_bytes':kernel.n_spills*4,'shared_bytes':kernel.metadata.shared}
        else:
            bindings=bindings or compiler();cu,_=bindings;_,fn,info=compile_kernel(shape,mode,bindings)
            if family=='wide':rows,cols,groups,_,_=config
            else:rows,cols,_,_=config;groups=k//256
            values=[ctypes.c_void_p(v.data_ptr()) for v in (x,w,r,s,out)];pointers=(ctypes.c_void_p*5)(*(ctypes.addressof(v) for v in values))
            _check(cu.cuLaunchKernel(fn,(n+cols-1)//cols,(m+rows-1)//rows,1,16*cols*groups,1,1,0,
                cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return (out,info) if resources else out

"""Research-only FP32 LayerNorm preserving PyTorch 2.8's logical Welford tree.

Arithmetic/layout reference: aten/src/ATen/native/cuda/layer_norm_kernel.cu,
PyTorch v2.8.0. Native logical threads own vectors of four values; this probe
maps four logical warps either to physical warps or to register arrays.
"""
import ctypes
import torch
from cuda.bindings import driver as cu,nvrtc
from benchmarks.small_vector_cuda import check

SOURCE=r'''
struct Stats {float mean,var,count;};
__device__ __forceinline__ Stats online(float x,Stats a){
    float d=x-a.mean,c=a.count+1.f;
    float m=a.mean+d*(1.f/c);
    return {m,a.var+d*(x-m),c};
}
__device__ __forceinline__ Stats merge(Stats current,Stats remote){
    float d=current.mean-remote.mean,c=current.count+remote.count;
    if(c==0.f)return {0.f,0.f,0.f};
    float inv=1.f/c,ra=remote.count*inv,rb=current.count*inv;
    return {ra*remote.mean+rb*current.mean,
            remote.var+current.var+d*d*remote.count*rb,c};
}
__device__ __forceinline__ float fixed_mean(float current,float remote,float ra,float rb){
    #if FIXED==2
    return __fmaf_rn(rb,current,__fmul_rn(ra,remote));
    #elif FIXED==3
    return __fmaf_rn(ra,remote,__fmul_rn(rb,current));
    #else
    return ra*remote+rb*current;
    #endif
}
template<int A,int B> __device__ __forceinline__ Stats fixed_merge(Stats current,Stats remote){
    constexpr float inv=1.f/float(A+B),ra=float(B)*inv,rb=float(A)*inv;
    float d=current.mean-remote.mean;
    return {fixed_mean(current.mean,remote.mean,ra,rb),
            remote.var+current.var+d*d*float(B)*rb,float(A+B)};
}
template<int COUNT> __device__ __forceinline__ Stats fixed_stats(const float* x,int logical){
    Stats s={};
    #pragma unroll
    for(int step=0;step<COUNT/4;++step){
        float4 v=reinterpret_cast<const float4*>(x)[logical+step*128];
        float values[4]={v.x,v.y,v.z,v.w};
        #pragma unroll
        for(int j=0;j<4;++j){
            float d=values[j]-s.mean;
            float m=s.mean+d*(1.f/float(step*4+j+1));
            s.var=s.var+d*(values[j]-m);s.mean=m;
        }
    }
    #pragma unroll
    for(int i=0;i<5;++i){
        int offset=16>>i;
        Stats other={__shfl_down_sync(0xffffffff,s.mean,offset),
                     __shfl_down_sync(0xffffffff,s.var,offset),0.f};
        // Counts double at each native shuffle level. Compiler unrolling makes
        // each coefficient constant without reordering the floating operations.
        float count=float(COUNT*(1<<i));
        float d=s.mean-other.mean;
        s={fixed_mean(s.mean,other.mean,.5f,.5f),other.var+s.var+d*d*count*.5f,count*2.f};
    }
    return s;
}
extern "C" __global__ void layer_norm(const float* __restrict__ X,
    const float* G,const float* B,float* Y,float* Mean,float* Rstd,float eps,int rows){
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int row=REGISTER ? blockIdx.x*(THREADS/32)+warp : blockIdx.x;
    if(row>=rows)return;
    constexpr int SLOTS=REGISTER?4:1;
    Stats st[SLOTS]={};
    #pragma unroll
    for(int slot=0;slot<SLOTS;++slot){
        int logical=lane+32*(REGISTER?slot:warp);
        #if FIXED
        if((REGISTER?slot:warp)<2)st[slot]=fixed_stats<((N/4+127)/128)*4>(X+row*N,logical);
        else st[slot]=fixed_stats<(N/4/128)*4>(X+row*N,logical);
        #else
        #pragma unroll UNROLL
        for(int idx=logical;idx<N/4;idx+=128){
            float4 value=reinterpret_cast<const float4*>(X+row*N)[idx];
            st[slot]=online(value.x,st[slot]);st[slot]=online(value.y,st[slot]);
            st[slot]=online(value.z,st[slot]);st[slot]=online(value.w,st[slot]);
        }
        #pragma unroll
        for(int offset=16;offset;offset/=2){
            Stats other={__shfl_down_sync(0xffffffff,st[slot].mean,offset),
                         __shfl_down_sync(0xffffffff,st[slot].var,offset),
                         __shfl_down_sync(0xffffffff,st[slot].count,offset)};
            st[slot]=merge(st[slot],other);
        }
        #endif
    }
    float mean,variance;
    if(REGISTER){
        if(lane==0){
            #if FIXED
            constexpr int H=((N/4+127)/128)*128, L=(N/4/128)*128;
            st[0]=fixed_merge<H,L>(st[0],st[2]);st[1]=fixed_merge<H,L>(st[1],st[3]);
            st[0]=fixed_merge<H+L,H+L>(st[0],st[1]);
            #else
            st[0]=merge(st[0],st[2]);st[1]=merge(st[1],st[3]);st[0]=merge(st[0],st[1]);
            #endif
        }
        mean=__shfl_sync(0xffffffff,st[0].mean,0);
        variance=__fdiv_rn(__shfl_sync(0xffffffff,st[0].var,0),float(N));
    }else{
        __shared__ Stats shared[4];
        if(lane==0)shared[warp]=st[0];
        __syncthreads();
        if(threadIdx.x==0){
            #if FIXED
            constexpr int H=((N/4+127)/128)*128, L=(N/4/128)*128;
            Stats a=fixed_merge<H,L>(shared[0],shared[2]),b=fixed_merge<H,L>(shared[1],shared[3]);
            shared[0]=fixed_merge<H+L,H+L>(a,b);
            #else
            Stats a=merge(shared[0],shared[2]),b=merge(shared[1],shared[3]);
            shared[0]=merge(a,b);
            #endif
        }
        __syncthreads();
        mean=shared[0].mean;variance=__fdiv_rn(shared[0].var,float(N));
    }
    float rs=rsqrtf(variance+eps);
    int start=REGISTER?lane:threadIdx.x,stride=REGISTER?32:128;
    #pragma unroll UNROLL
    for(int idx=start;idx<N/4;idx+=stride){
        float4 v=reinterpret_cast<const float4*>(X+row*N)[idx];
        float4 g=reinterpret_cast<const float4*>(G)[idx],b=reinterpret_cast<const float4*>(B)[idx],out;
        out.x=g.x*(rs*(v.x-mean))+b.x;out.y=g.y*(rs*(v.y-mean))+b.y;
        out.z=g.z*(rs*(v.z-mean))+b.z;out.w=g.w*(rs*(v.w-mean))+b.w;
        reinterpret_cast<float4*>(Y+row*N)[idx]=out;
    }
    if((REGISTER&&lane==0)||(!REGISTER&&threadIdx.x==0)){Mean[row]=mean;Rstd[row]=rs;}
}
'''
CACHE={}


def compile_kernel(n,register,threads,unroll,fmad,fixed=False):
    key=(n,register,threads,unroll,fmad,fixed)
    if key in CACHE:return CACHE[key]
    source=('\n'.join(f'#define {k} {v}' for k,v in dict(N=n,REGISTER=int(register),THREADS=threads,UNROLL=unroll,FIXED=int(fixed)).items())+'\n'+SOURCE).encode()
    prog=check(nvrtc.nvrtcCreateProgram(source,b'layer_norm.cu',0,[],[]))
    opts=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad='+str(fmad).lower().encode()]
    result=nvrtc.nvrtcCompileProgram(prog,len(opts),opts)
    size=check(nvrtc.nvrtcGetProgramLogSize(prog));log=b' '*size;check(nvrtc.nvrtcGetProgramLog(prog,log))
    if int(result[0]):raise RuntimeError(log.decode())
    size=check(nvrtc.nvrtcGetCUBINSize(prog));blob=b' '*size;check(nvrtc.nvrtcGetCUBIN(prog,blob))
    check(nvrtc.nvrtcDestroyProgram(prog));module=check(cu.cuModuleLoadData(blob));fn=check(cu.cuModuleGetFunction(module,b'layer_norm'))
    attr=cu.CUfunction_attribute
    res={name:check(cu.cuFuncGetAttribute(a,fn)) for name,a in [('registers',attr.CU_FUNC_ATTRIBUTE_NUM_REGS),('local_bytes',attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),('shared_bytes',attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)]}
    CACHE[key]=(module,fn,res);return CACHE[key]


def layer_norm(x,g,b,eps=1e-5,config=(True,128,1,True),resources=False):
    register,threads,unroll,fmad,*extra=config;fixed=extra[0] if extra else False
    n=x.shape[-1];rows=x.numel()//n
    if (x.dtype!=torch.float32 or not x.is_cuda or n not in (768,1280)
            or not x.is_contiguous() or g.shape!=(n,) or b.shape!=(n,)
            or any(t.dtype!=x.dtype or t.device!=x.device or not t.is_contiguous() or t.data_ptr()%16 for t in (x,g,b))
            or threads not in (32,64,128,256) or (not register and threads!=128) or unroll not in (1,4)):
        raise ValueError('Expected aligned contiguous FP32 native LayerNorm operands')
    _,fn,res=compile_kernel(n,register,threads,unroll,fmad,fixed)
    y=torch.empty(x.shape,device=x.device,dtype=x.dtype);mean=torch.empty(rows,device=x.device);rs=torch.empty_like(mean)
    values=[ctypes.c_void_p(t.data_ptr()) for t in (x,g,b,y,mean,rs)]+[ctypes.c_float(eps),ctypes.c_int(rows)]
    pointers=(ctypes.c_void_p*len(values))(*(ctypes.addressof(v) for v in values))
    grid=(rows+threads//32-1)//(threads//32) if register else rows
    check(cu.cuLaunchKernel(fn,grid,1,1,threads,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return (y,mean,rs,res) if resources else (y,mean,rs)

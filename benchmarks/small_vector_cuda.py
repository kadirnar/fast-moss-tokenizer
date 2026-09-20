"""Research-only NVRTC FP32 tiled cyclic GEMM with vectorized per-thread lanes."""
import ctypes
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cu, nvrtc
from triton.testing import do_bench_cudagraph
from benchmarks.compare import difference
from fast_moss.loading import strict_precision,REVISION

SOURCE=r'''
extern "C" __global__ void small(const float* __restrict__ X,
                                const float* __restrict__ W,float* __restrict__ Y){
  constexpr int LANES=16/VEC;
  int lane=threadIdx.x%LANES;
  int n=blockIdx.x*(THREADS/LANES)+threadIdx.x/LANES;
  int m=blockIdx.y*ROWS;
  __shared__ __align__(16) float sx[SHARED ? ROWS*256 : 1];
  float total[ROWS][VEC]={};
  for(int tile=0;tile<(K+255)/256;++tile){
    if(SHARED){
      for(int i=threadIdx.x;i<ROWS*256;i+=THREADS){
        int r=i/256,k=tile*256+i%256;
        sx[i]=(m+r<M && k<K)?X[(m+r)*K+k]:0.f;
      }
      __syncthreads();
    }
    float acc[ROWS][VEC]={};
    #pragma unroll UNROLL
    for(int step=0;step<16;++step){
      int local=step*16+lane*VEC,k=tile*256+local;
      float w[VEC]={};
      if(n<N && k<K){
        #if VEC==4
        float4 v=*reinterpret_cast<const float4*>(W+n*K+k);
        w[0]=v.x;w[1]=v.y;w[2]=v.z;w[3]=v.w;
        #elif VEC==2
        float2 v=*reinterpret_cast<const float2*>(W+n*K+k);
        w[0]=v.x;w[1]=v.y;
        #else
        w[0]=W[n*K+k];
        #endif
      }
      #pragma unroll
      for(int r=0;r<ROWS;++r){
        float x[VEC]={};
        if(m+r<M && k<K){
          const float* xp=SHARED?sx+r*256+local:X+(m+r)*K+k;
          #if VEC==4
          float4 v=*reinterpret_cast<const float4*>(xp);
          x[0]=v.x;x[1]=v.y;x[2]=v.z;x[3]=v.w;
          #elif VEC==2
          float2 v=*reinterpret_cast<const float2*>(xp);
          x[0]=v.x;x[1]=v.y;
          #else
          x[0]=*xp;
          #endif
        }
        #pragma unroll
        for(int v=0;v<VEC;++v)acc[r][v]=__fmaf_rn(x[v],w[v],acc[r][v]);
      }
    }
    #pragma unroll
    for(int r=0;r<ROWS;++r){
      #pragma unroll
      for(int v=0;v<VEC;++v)total[r][v]=__fadd_rn(total[r][v],acc[r][v]);
    }
    if(SHARED)__syncthreads();
  }
  #pragma unroll
  for(int r=0;r<ROWS;++r){
    float out=__shfl_sync(0xffffffff,total[r][0],0,LANES);
    #pragma unroll
    for(int i=1;i<16;++i){
      float value=__shfl_sync(0xffffffff,total[r][i%VEC],i/VEC,LANES);
      out=__fadd_rn(out,value);
    }
    if(lane==0 && n<N && m+r<M)Y[(m+r)*N+n]=out;
  }
}
'''

CACHE={}

def check(result):
    if int(result[0]):raise RuntimeError(str(result[0]))
    return result[1] if len(result)==2 else result[1:]


def compile_kernel(shape,config):
    key=(shape,config)
    if key in CACHE:return CACHE[key]
    m,n,k=shape;rows,threads,vec,shared,unroll=config
    definitions=dict(M=m,N=n,K=k,ROWS=rows,THREADS=threads,VEC=vec,SHARED=int(shared),UNROLL=unroll)
    source=('\n'.join(f'#define {name} {value}' for name,value in definitions.items())+'\n'+SOURCE).encode()
    program=check(nvrtc.nvrtcCreateProgram(source,b'small.cu',0,[],[]))
    options=[b'--gpu-architecture=sm_120',b'--std=c++17',b'--ftz=false',b'--fmad=false']
    result=nvrtc.nvrtcCompileProgram(program,len(options),options)
    size=check(nvrtc.nvrtcGetProgramLogSize(program));log=b' '*size
    check(nvrtc.nvrtcGetProgramLog(program,log))
    if int(result[0]):raise RuntimeError(log.decode())
    size=check(nvrtc.nvrtcGetCUBINSize(program));blob=b' '*size
    check(nvrtc.nvrtcGetCUBIN(program,blob));check(nvrtc.nvrtcDestroyProgram(program))
    module=check(cu.cuModuleLoadData(blob));fn=check(cu.cuModuleGetFunction(module,b'small'))
    attr=cu.CUfunction_attribute
    metadata={name:check(cu.cuFuncGetAttribute(value,fn)) for name,value in
              [('registers',attr.CU_FUNC_ATTRIBUTE_NUM_REGS),('shared_bytes',attr.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
               ('local_bytes',attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)]}
    CACHE[key]=(module,fn,metadata)
    return CACHE[key]


def vector(x,w,config):
    m,k=x.shape;n=w.shape[0];rows,threads,vec,shared,unroll=config
    _,fn,_=compile_kernel((m,n,k),tuple(config))
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    values=[ctypes.c_void_p(t.data_ptr()) for t in (x,w,out)]
    pointers=(ctypes.c_void_p*3)(*(ctypes.addressof(v) for v in values))
    cols=threads//(16//vec)
    check(cu.cuLaunchKernel(fn,(n+cols-1)//cols,(m+rows-1)//rows,1,threads,1,1,0,
        cu.CUstream(torch.cuda.current_stream().cuda_stream),ctypes.addressof(pointers),0))
    return out

@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research-only explicit CUDA row reuse/vectorized load search, warm components',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'nvrtc_version':list(check(nvrtc.nvrtcVersion())),'records':[]}
    for shape in [(6,3072,768),(6,768,3072),(3,3840,1280),(12,2304,768)]:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        record={'shape':shape,'native_ms':do_bench_cudagraph(lambda:F.linear(x,w),rep=20),'trials':[]}
        rows=3 if shape[0]==3 else 6
        for threads in [64,128,256]:
            for vec in [1,2,4]:
                for shared in [False,True]:
                    for unroll in [4,16]:
                        config=(rows,threads,vec,shared,unroll)
                        fn=lambda:vector(x,w,config)
                        d=difference(ref,fn());ms=do_bench_cudagraph(fn,rep=10)
                        record['trials'].append({'config':config,'difference':d,'ms':ms,
                                                 'resources':compile_kernel(shape,config)[2]})
        report['records'].append(record)
        print(shape,'native',record['native_ms'],'best',min(record['trials'],key=lambda t:t['ms']),flush=True)
        Path('results/small_vector_cuda.json').write_text(json.dumps(report,indent=2)+'\n')
    for module,_,_ in CACHE.values():check(cu.cuModuleUnload(module))
    CACHE.clear()

if __name__=='__main__':main()

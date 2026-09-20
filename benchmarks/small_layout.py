"""Research-only explicit SIMT layouts with register-resident row reuse."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.testing import do_bench_cudagraph
from fast_moss import small_matrices as sm
from fast_moss.loading import strict_precision,REVISION
from fast_moss.graphs import GraphedCallable
from benchmarks.ffn_resources import resources
from benchmarks.matrices import evicted_replay


@g.jit
def _layout(X,W,Y,M:gl.constexpr,N:gl.constexpr,K:gl.constexpr,
            BM:gl.constexpr,BN:gl.constexpr,WARPS:gl.constexpr,UNROLL:gl.constexpr):
    layout:gl.constexpr=gl.BlockedLayout([1,1,1],[1,2,16],[1,WARPS,1],[2,1,0])
    rm=gl.program_id(1)*BM+gl.arange(0,BM,layout=gl.SliceLayout(1,gl.SliceLayout(2,layout)))
    rn=gl.program_id(0)*BN+gl.arange(0,BN,layout=gl.SliceLayout(0,gl.SliceLayout(2,layout)))
    lane=gl.arange(0,16,layout=gl.SliceLayout(0,gl.SliceLayout(1,layout)))
    total=gl.full((BM,BN,16),0,gl.float32,layout)
    for tile in range(triton.cdiv(K,256)):
        acc=gl.full((BM,BN,16),0,gl.float32,layout)
        for step in tl.range(16,loop_unroll_factor=UNROLL):
            k=tile*256+step*16+lane
            x=gl.load(X+rm[:,None,None]*K+k[None,None,:],(rm[:,None,None]<M)&(k[None,None,:]<K),gl.full((BM,1,16),0,gl.float32,layout))
            w=gl.load(W+rn[None,:,None]*K+k[None,None,:],(rn[None,:,None]<N)&(k[None,None,:]<K),gl.full((1,BN,16),0,gl.float32,layout))
            acc=gl.fma(x,w,acc)
        total=total+acc
    # Only lane zero writes. Its XOR partner i supplies native lane i.
    out=total
    for i in tl.static_range(1,16):
        v=gl.inline_asm_elementwise('shfl.sync.bfly.b32 $0, $1, $2, 31, -1;',
            constraints='=f,f,r',args=[total,gl.full((BM,BN,16),i,gl.int32,layout)],dtype=gl.float32,is_pure=True,pack=1)
        out=out+v
    gl.store(Y+rm[:,None,None]*N+rn[None,:,None]+gl.full((1,1,16),0,gl.int32,layout),
             out,(rm[:,None,None]<M)&(rn[None,:,None]<N)&(lane[None,None,:]==0))


def run(x,w,config,return_resources=False):
    m,k=x.shape;n=w.shape[0];bm,bn,warps,u=config
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    kernel=_layout[(triton.cdiv(n,bn),triton.cdiv(m,bm))](x,w,out,m,n,k,bm,bn,warps,u,
                num_warps=warps,enable_fp_fusion=False)
    return (out,resources(kernel)) if return_resources else out


@torch.inference_mode()
def main():
    strict_precision();cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    report={'scope':'research explicit SIMT layouts and wider row reuse; original FP32',
            'revision':REVISION,'previous_commit':'fd559c3','torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'records':[]}
    flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
    shapes=[(8,768,768),(8,1280,5120),(16,768,768),(16,2304,768),(16,3072,768),(8,768,240)]
    for shape in shapes:
        x,w=cases[shape]['x'],cases[shape]['weight'];ref=F.linear(x,w)
        row={'shape':shape,'previous_config':sm.CONFIGS[shape],'previous_ms':do_bench_cudagraph(lambda:sm.linear(x,w),rep=20),'trials':[]}
        for bm in [4,8,16]:
            for bn,warps in [(2,1),(4,1),(4,2),(8,2)]:
                for u in [4,16]:
                    cfg=(bm,bn,warps,u)
                    out,res=run(x,w,cfg,True);exact=torch.equal(out.view(torch.int32),ref.view(torch.int32))
                    graph=GraphedCallable(lambda z:(run(z,w,cfg),),x)
                    warm=do_bench_cudagraph(lambda:run(x,w,cfg),rep=20)
                    cold=evicted_replay(graph,flush,repeats=25);del graph
                    row['trials'].append({'config':cfg,'bits_equal':exact,'resources':res,'warm_ms':warm,'cold':cold})
        report['records'].append(row)
        best=min(row['trials'],key=lambda t:t['warm_ms'])
        print(shape,'previous',row['previous_ms'],'best',best,flush=True)
        Path('results/small_layout.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

"""Research-only ordered reduction fused with GELU or a residual update."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.testing import do_bench_cudagraph
from fast_moss.ordered_matrices import _partials, linear as baseline
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable
from benchmarks.compare import difference


@triton.jit
def mul_rn(x,y):
    return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,y],
                                     dtype=tl.float32,is_pure=True,pack=1)


@triton.jit
def add_rn(x,y):
    return tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[x,y],
                                     dtype=tl.float32,is_pure=True,pack=1)


@triton.jit
def _epilogue(P, X, S, Y, NUMEL:tl.constexpr, N:tl.constexpr, PARTS:tl.constexpr,
               MODE:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    acc=tl.load(P+i,i<NUMEL,0)
    for part in range(1,PARTS):
        acc=add_rn(acc,tl.load(P+part*NUMEL+i,i<NUMEL,0))
    if MODE=='gelu':
        half=mul_rn(acc,0.5)
        erf=libdevice.erf(mul_rn(acc,0.7071067811865476))
        acc=mul_rn(half,add_rn(erf,1.0))
    elif MODE=='residual':
        residual=tl.load(X+i,i<NUMEL,0)
        scale=tl.load(S+i%N,i<NUMEL,0)
        acc=add_rn(residual,mul_rn(acc,scale))
    tl.store(Y+i,acc,i<NUMEL)


def epilogue(partials,mode,residual=None,scale=None,fusion=True,math_library=None):
    parts,m,n=partials.shape
    out=torch.empty((m,n),device=partials.device,dtype=torch.float32)
    kwargs={} if math_library is None else {'extern_libs':{'libdevice':math_library}}
    _epilogue[(triton.cdiv(m*n,256),)](partials,residual if residual is not None else partials,
        scale if scale is not None else partials,out,m*n,n,parts,mode,256,enable_fp_fusion=fusion,**kwargs)
    return out


def linear(x,packed,mode,residual=None,scale=None,fusion=True,stages=3,math_library=None):
    m,k=x.shape;n=packed.shape[1]
    partials=torch.empty((k//256,m,n),device=x.device,dtype=x.dtype)
    _partials[(1,n//128,k//256)](x,packed,partials,m,n,k,256,32,128,32,num_warps=4,
                                    num_stages=stages,enable_fp_fusion=False)
    return epilogue(partials,mode,residual,scale,fusion,math_library)


@torch.inference_mode()
def main():
    strict_precision();torch.manual_seed(745)
    report={'scope':'research-only exact GELU/residual epilogue probe','torch':torch.__version__,
            'triton':triton.__version__,'gpu':torch.cuda.get_device_name(),'gelu':[],'matrices':[]}
    values=[('ramp',torch.linspace(-15,15,1048576,device='cuda')),
            ('random',torch.randn(1048576,device='cuda')*10),
            ('subnormal',torch.randn(1048576,device='cuda')*1e-38),
            ('tiny',torch.randn(1048576,device='cuda')*1e-20),
            ('large',torch.randn(1048576,device='cuda')*1e20)]
    for fusion in [False,True]:
        for label,value in values:
            ref=F.gelu(value).view(1,-1)
            out=epilogue(value.view(1,1,-1),'gelu',fusion=fusion)
            r={'input':label,'fusion':fusion,'difference':difference(ref,out),
               'bitwise':torch.equal(ref.view(torch.int32),out.view(torch.int32))}
            report['gelu'].append(r);print(r,flush=True)
    cases=torch.load('results/matrix_inputs.pt',weights_only=True)
    for shape,mode in [((24,5120,1280),'gelu'),((24,1280,5120),'residual')]:
        x,w=cases[shape]['x'],cases[shape]['weight'];packed=w.T.contiguous()
        residual=torch.randn(24,shape[1],device='cuda');scale=torch.randn(shape[1],device='cuda')
        native=(lambda z:F.gelu(baseline(z,packed))) if mode=='gelu' else (lambda z:residual+baseline(z,packed)*scale)
        ref=native(x)
        for fusion in [False,True]:
            fn=lambda z:linear(z,packed,mode,residual,scale,fusion)
            out=fn(x);graph=GraphedCallable(lambda z:(fn(z),),x)
            r={'shape':shape,'mode':mode,'fusion':fusion,'eager':difference(ref,out),
               'graph':difference(ref,graph(x)[0]),
               'native_ms':do_bench_cudagraph(lambda:native(x),rep=30),
               'candidate_ms':do_bench_cudagraph(lambda:fn(x),rep=30)}
            report['matrices'].append(r);print(r,flush=True);del graph
    report['all_exact']=all(r['difference']['exact'] for r in report['gelu']) and all(r[k]['exact'] for r in report['matrices'] for k in ['eager','graph'])
    Path('results/ordered_epilogue.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

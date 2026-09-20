"""Exact native FP32 epilogues, including signed-zero and graph input updates."""
import pytest
import torch
import torch.nn.functional as F
from benchmarks.gemv_epilogue import linear as research_linear
from fast_moss.ffn import math_library,gemv_linear
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable


@pytest.mark.parametrize('mode,n,k',[('gelu',5120,1280),('residual',1280,5120)])
@pytest.mark.parametrize('case',['random','subnormal','positive_underflow','signed_zero'])
@pytest.mark.parametrize('backend',['research','runtime'])
@torch.inference_mode()
def test_native_epilogues(mode,n,k,case,backend):
    linear=research_linear if backend=='research' else gemv_linear
    strict_precision();torch.manual_seed(72419);library=math_library()
    w=torch.randn(n,k,device='cuda')*.02
    x=torch.randn(1,k,device='cuda')
    residual=torch.randn(1,n,device='cuda');scale=torch.randn(n,device='cuda')*.01
    if case=='subnormal':x*=1e-38
    elif case=='positive_underflow':w.fill_(.125);x.fill_(-1.401298464324817e-45);residual.zero_();scale.fill_(1.)
    elif case=='signed_zero':x.fill_(-0.);residual.fill_(-0.);scale.fill_(-1.)
    def reference(z):
        y=F.linear(z,w)
        return F.gelu(y) if mode=='gelu' else residual+y*scale
    graph=GraphedCallable(lambda z:(linear(z,w,mode,residual,scale,library),),x)
    for z in (x,-x,x*.17):
        ref=reference(z).view(torch.int32)
        assert torch.equal(linear(z,w,mode,residual,scale,library).view(torch.int32),ref)
        assert torch.equal(graph(z)[0].view(torch.int32),ref)


@torch.inference_mode()
@pytest.mark.parametrize('backend',['research','runtime'])
def test_epilogue_rejects_invalid_operands(backend):
    linear=research_linear if backend=='research' else gemv_linear
    w=torch.randn(1280,5120,device='cuda');x=torch.randn(1,5120,device='cuda')
    r=torch.randn(1,1280,device='cuda');s=torch.ones(1280,device='cuda')
    for a,b,mode,residual,scale in [(x,w,'bad',r,s),(x.double(),w,'residual',r,s),
        (x,w,'residual',None,s),(x,w,'residual',r,s.double()),
        (x,w,'residual',r[:,:1279],s),(x[:,::2],w[:,::2],'gelu',None,None)]:
        with pytest.raises(ValueError):linear(a,b,mode,residual,scale,math_library())

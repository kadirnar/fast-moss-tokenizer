"""Native Welford tree and affine-output gates for the research CUDA norms."""
import pytest
import torch
from benchmarks.native_layer_norm import layer_norm
from benchmarks.native_layer_norm_tune import bits
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@pytest.mark.parametrize('n',[768,1280])
@pytest.mark.parametrize('rows',[1,7,65])
@pytest.mark.parametrize('register',[False,True])
@pytest.mark.parametrize('eps',[0.,1e-5])
@torch.inference_mode()
def test_fixed_counts_native_statistics_and_graph(n,rows,register,eps):
    strict_precision();torch.manual_seed(93715)
    x=torch.randn(rows,n,device='cuda');g=torch.randn(n,device='cuda');b=torch.randn(n,device='cuda')
    cfg=(register,128,1,True,3)
    graph=GraphedCallable(lambda z:layer_norm(z,g,b,eps,cfg),x)
    for z in (x,x*1e-20,x*.01+1e4,torch.full_like(x,-0.)):
        ref=torch.native_layer_norm(z,(n,),g,b,eps)
        assert all(bits(a,v) for a,v in zip(ref,layer_norm(z,g,b,eps,cfg)))
        assert all(bits(a,v) for a,v in zip(ref,graph(z)))


@torch.inference_mode()
def test_layer_norm_operand_guards():
    x=torch.randn(2,768,device='cuda');g=torch.randn(768,device='cuda');b=torch.randn_like(g)
    offset=torch.empty(x.numel()+1,device='cuda')[1:].view_as(x)
    for a,weight,bias,cfg in [(x.double(),g,b,(False,128,1,True,3)),
        (x.cpu(),g.cpu(),b.cpu(),(False,128,1,True,3)),(offset,g,b,(False,128,1,True,3)),
        (x[:,::2],g[::2],b[::2],(False,128,1,True,3)),(x,g,b,(False,64,1,True,3))]:
        with pytest.raises(ValueError):layer_norm(a,weight,bias,config=cfg)

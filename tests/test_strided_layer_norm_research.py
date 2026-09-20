"""Exact strided loads, shared-transpose boundaries and graph regressions."""
import pytest
import torch
from benchmarks.strided_layer_norm import layer_norm
from benchmarks.native_layer_norm_tune import bits
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@torch.inference_mode()
@pytest.mark.parametrize('n',[768,1280])
@pytest.mark.parametrize('batch,time',[(1,3),(2,6),(3,2)])
@pytest.mark.parametrize('config',[(False,128,1,False,0),(True,128,4,False,0),
                                  (True,256,4,True,0),(True,256,1,True,4)])
def test_strided_welford_partial_tiles_signed_zero_and_graph(n,batch,time,config):
    strict_precision();torch.manual_seed(951)
    x=torch.randn(batch,n,time,device='cuda').transpose(1,2)
    g=torch.randn(n,device='cuda');b=torch.randn_like(g)
    for eps in (0.,1e-5):
        fn=lambda z:layer_norm(z,g,b,eps,config)
        graph=GraphedCallable(fn,x)
        for z in (x,x*1e-20,x*.01+1e4,torch.full_like(x,-0.),torch.full_like(x,float('inf'))):
            ref=torch.native_layer_norm(z,(n,),g,b,eps)
            assert all(bits(a,v) for a,v in zip(ref,fn(z)))
            assert all(bits(a,v) for a,v in zip(ref,graph(z)))


@torch.inference_mode()
def test_strided_operand_and_launch_guards():
    x=torch.randn(2,768,3,device='cuda').transpose(1,2);g=torch.ones(768,device='cuda');b=torch.zeros_like(g)
    cfg=(False,128,1,False,0)
    for z,w,bias,c in [(x.cpu(),g.cpu(),b.cpu(),cfg),(x.double(),g.double(),b.double(),cfg),
        (x.contiguous(),g,b,cfg),(x[:,:,::2],g[::2],b[::2],cfg),
        (x,g,b,(False,128,1,True,0)),(x,g,b,(True,128,4,True,1))]:
        with pytest.raises(ValueError,match='dense transposed'):layer_norm(z,w,bias,config=c)

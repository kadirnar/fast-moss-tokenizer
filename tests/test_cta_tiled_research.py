"""Native tile/lane order, half-warp masks and partial row tiles."""
import pytest
import torch
import torch.nn.functional as F
from benchmarks.cta_tiled_matrix import linear as cuda
from benchmarks.cta_tiled_triton import linear as triton
from benchmarks.cta_tiled_tune import SHAPES
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('backend,config',[(cuda,(1,1,4,False)),(cuda,(3,2,16,True)),
    (cuda,(6,8,16,False)),(triton,(4,2,2,16)),(triton,(8,8,4,4))])
def test_native_arithmetic_partial_tiles_half_warps_and_graph(shape,backend,config):
    strict_precision();torch.manual_seed(3674)
    m,n,k=shape;x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')*.03
    fn=lambda a,b:(backend(a,b,config),)
    graph=GraphedCallable(fn,x,w)
    for a,b in [(x,w),(-x,w),(x*1e-38,w),(x*1e20,w),(torch.full_like(x,-0.),w),
                (torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]:
        ref=F.linear(a,b)
        assert bits(fn(a,b)[0],ref) and bits(graph(a,b)[0],ref)


@torch.inference_mode()
@pytest.mark.parametrize('backend,config',[(cuda,(4,4,16,True)),(triton,(4,4,4,16))])
def test_operand_and_launch_guards(backend,config):
    x=torch.randn(4,768,device='cuda');w=torch.randn(768,768,device='cuda')
    for a,b in [(x.cpu(),w.cpu()),(x.double(),w.double()),(x[:,::2],w[:,::2]),(x[:0],w),(x,w[:0]),(x,w.T)]:
        with pytest.raises(ValueError,match='contiguous CUDA FP32'):backend(a,b,config)
    with pytest.raises(ValueError,match='launch dimensions'):backend(x,w,(0,0,0,False))

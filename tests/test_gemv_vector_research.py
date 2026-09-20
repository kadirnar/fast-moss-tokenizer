"""Native reduction, vector alignment, and prefetch gates for the CUDA prototype."""
import pytest
import torch
import torch.nn.functional as F

from benchmarks.gemv_vector import gemv
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@pytest.mark.parametrize('shape',[(768,1280,32),(1280,768,16),(3840,1280,8)])
@pytest.mark.parametrize('vec,prefetch',[(1,4),(2,2),(4,4),(8,2)])
@torch.inference_mode()
def test_vector_prefetch_native_order_and_graph(shape,vec,prefetch):
    strict_precision();torch.manual_seed(94825)
    n,k,lanes=shape;cfg=(lanes,vec,32,prefetch,4)
    w=torch.randn(n,k,device='cuda')*.02
    inputs=[torch.randn(1,k,device='cuda'),torch.randn(1,k,device='cuda')*1e-38,
            torch.full((1,k),-0.,device='cuda')]
    graph=GraphedCallable(lambda z:(gemv(z,w,cfg),),inputs[0])
    for x in inputs:
        ref=F.linear(x,w).view(torch.int32)
        assert torch.equal(gemv(x,w,cfg).view(torch.int32),ref)
        assert torch.equal(graph(x)[0].view(torch.int32),ref)
    del graph
    w.fill_(.125);x=torch.full_like(inputs[0],-1.401298464324817e-45)
    graph=GraphedCallable(lambda z:(gemv(z,w,cfg),),x)
    ref=F.linear(x,w).view(torch.int32)
    assert torch.equal(gemv(x,w,cfg).view(torch.int32),ref)
    assert torch.equal(graph(x)[0].view(torch.int32),ref)


@torch.inference_mode()
def test_vector_input_contract_and_alignment():
    x=torch.randn(1,1280,device='cuda');w=torch.randn(768,1280,device='cuda')
    cfg=(32,4,32,2,4)
    misaligned=torch.empty(x.numel()+1,device='cuda')[1:].view_as(x)
    misaligned_w=torch.empty(w.numel()+1,device='cuda')[1:].view_as(w)
    invalid=[(misaligned,w,cfg),(x,misaligned_w,cfg),(x.double(),w,cfg),
             (x,w.T,cfg),(x.cpu(),w.cpu(),cfg),(x,w,(0,4,32,2,4)),
             (x,w,(32,3,32,2,4)),(x,w,(32,4,31,2,4)),
             (x[:,:1256].contiguous(),w[:,:1256].contiguous(),cfg)]
    for a,b,c in invalid:
        with pytest.raises(ValueError):gemv(a,b,c)

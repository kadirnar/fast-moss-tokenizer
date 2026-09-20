import pytest
import torch
import torch.nn.functional as F
from benchmarks.wide_cta_matrix import linear
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision

SHAPES=[(3,1280,5120),(4,768,3072),(8,768,1280),(8,768,3072),(12,768,3072)]

@torch.inference_mode()
@pytest.mark.parametrize('shape',SHAPES)
@pytest.mark.parametrize('rows,cols,groups,unroll,distribute',[
    (3,1,1,4,False),(3,1,2,16,True),(4,2,4,4,False),(4,2,4,16,True)])
def test_wide_shared_partitions(shape,rows,cols,groups,unroll,distribute):
    strict_precision();torch.manual_seed(1147);m,n,k=shape
    x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')*.02
    config=(rows,cols,groups,unroll,distribute)
    for a,b in [(x,w),(x*1e-38,w),(x*1e20,w),(torch.full_like(x,-0.),w),
                (torch.full_like(x,-1.401298464324817e-45),torch.full_like(w,.125))]:
        ref=F.linear(a,b);assert bits(linear(a,b,config),ref)
        # Output tails use the same arithmetic reference before dropping a column.
        assert bits(linear(a,b[:-1],config),ref[:,:-1].contiguous())
    from fast_moss.graphs import GraphedCallable
    graph=GraphedCallable(lambda z:(linear(z,w,config),),x)
    assert bits(graph(x)[0],F.linear(x,w))


def test_wide_launch_guards():
    with pytest.raises(ValueError,match='valid wide-CTA launch'):
        linear(torch.randn(4,3072),torch.randn(768,3072),(4,2,4,4,True))
    x=torch.empty((4,5120),device='cuda');w=torch.empty((1280,5120),device='cuda')
    for config in [(8,8,20,16,False),(4,1,21,4,False),(4,1,0,4,False)]:
        with pytest.raises(ValueError,match='valid wide-CTA launch'):linear(x,w,config)

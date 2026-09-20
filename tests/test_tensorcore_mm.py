"""Numerical and graph checks; these do not certify codec quality."""
import pytest
import torch
from benchmarks.tensorcore_mm import linear
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('mode',['tf32x3','bf16x6'])
@pytest.mark.parametrize('scale_a,scale_b',[(1.,1.),(1e-20,1e20),(20.,.05)])
def test_tensorcore_odd_tiles_and_exponent_range(mode,scale_a,scale_b):
    torch.manual_seed(372)
    x=torch.randn(37,129,device='cuda')*scale_a
    w=torch.randn(97,129,device='cuda')*scale_b
    flag=torch.backends.cuda.matmul.allow_tf32
    out=linear(x,w,(16,32,32,4,2),mode=mode)
    gold=(x.double()@w.double().T).float()
    torch.testing.assert_close(out,gold,atol=1e-4,rtol=1e-4)
    graph=GraphedCallable(lambda z:(linear(z,w,(16,32,32,4,2),mode=mode),),x)
    assert torch.equal(graph(x)[0],out)
    assert torch.backends.cuda.matmul.allow_tf32==flag
    del graph


def test_tensorcore_rejects_incompatible_storage():
    with pytest.raises(ValueError):linear(torch.zeros(2,4),torch.zeros(3,4))
    with pytest.raises(ValueError):linear(torch.zeros(2,4,device='cuda'),torch.zeros(3,4,device='cuda'),mode='tf32')

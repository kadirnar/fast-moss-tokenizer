import pytest
import torch
import torch.nn.functional as F
from benchmarks.attention_residual import CONFIGS,linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision

GEOMETRIES=[(shape,t) for shape in sorted(CONFIGS) for t in range(1,shape[0]+1) if shape[0]%t==0]

def bits(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('shape,time',GEOMETRIES)
def test_attention_residual_rounding_layout_and_graph(shape,time):
    strict_precision();torch.manual_seed(963)
    m,n,k=shape;x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')*.02
    r=torch.randn(m//time,n,time,device='cuda').transpose(1,2);s=torch.randn(n,device='cuda')*.1
    def fn(a,ww,rr,ss):return (linear(a,ww,rr,ss),)
    graph=GraphedCallable(fn,x,w,r,s)
    for a,ww,rr,ss in [(x,w,r,s),(x*1e-38,w,r*0,s),(x*1e10,w,r,s),
                       (torch.full_like(x,-0.),w,torch.full_like(r,-0.),s),
                       (torch.full_like(x,-1.401298464324817e-45),w.abs()*.01,torch.full_like(r,-0.),s),
                       (x,w,r,torch.zeros_like(s))]:
        ref=rr+F.linear(a,ww).reshape_as(rr)*ss
        for out in (fn(a,ww,rr,ss)[0],graph(a,ww,rr,ss)[0]):
            bits(out,ref);assert out.stride()==ref.stride()


def test_attention_residual_guards_and_default_dtype():
    with pytest.raises(ValueError,match='FP32'):linear(torch.empty(1,1280),torch.empty(1280,1280),torch.empty(1,1,1280),torch.empty(1280))
    x=torch.zeros(1,1280,device='cuda');w=torch.zeros(1280,1280,device='cuda');r=torch.zeros(1,1,1280,device='cuda');s=torch.zeros(1280,device='cuda')
    original=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert linear(x,w,r,s).dtype==torch.float32
    finally:torch.set_default_dtype(original)

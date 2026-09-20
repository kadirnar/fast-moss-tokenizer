import pytest
import torch
import torch.nn.functional as F
from benchmarks.strided_ffn import CONFIGS,linear
from fast_moss.ffn import math_library
from fast_moss.normalization import compiler
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from benchmarks.native_layer_norm_tune import bits

GEOMETRIES=[(b,m//b,n) for m,n,k in sorted(CONFIGS) if n<k for b in range(1,m) if m%b==0]

@torch.inference_mode()
@pytest.mark.parametrize('batch,time,width',GEOMETRIES)
def test_dense_transposed_residual_rounding_layout_and_graph(batch,time,width):
    strict_precision();torch.manual_seed(384)
    m=batch*time;k=width*4
    x=torch.randn(m,k,device='cuda');w=torch.randn(width,k,device='cuda')*.02
    r=torch.randn(batch,width,time,device='cuda').transpose(1,2);s=torch.randn(width,device='cuda')
    library=math_library();bindings=compiler()
    fn=lambda a,b,c:(linear(a,w,b,c,library,bindings),)
    graph=GraphedCallable(fn,x,r,s)
    for factor in (0.,1.,-1.,1e-38,1e-20,1e20):
        a=x*factor;raw=F.linear(a,w).reshape(batch,time,width);ref=r+raw*s
        eager=fn(a,r,s)[0];captured=graph(a,r,s)[0]
        assert bits(eager,ref) and bits(captured,ref)
        assert eager.stride()==captured.stride()==ref.stride()==r.stride()
    raw=F.linear(x,w).reshape(batch,time,width)
    cancel=torch.empty_like(r);cancel.copy_(-raw*s);ref=cancel+raw*s
    assert bits(fn(x,cancel,s)[0],ref) and bits(graph(x,cancel,s)[0],ref)
    w.fill_(.125);minimum=torch.full_like(x,-1.401298464324817e-45)
    zeros=torch.full_like(r,-0.);ones=torch.ones_like(s);ref=zeros+F.linear(minimum,w).reshape_as(r)*ones
    assert bits(linear(minimum,w,zeros,ones,library,bindings),ref)


def test_strided_helper_guards():
    with pytest.raises(ValueError,match='contiguous FP32'):linear(torch.randn(2,3072),torch.randn(768,3072),None,None)
    x=torch.empty(2,3072,device='cuda');w=torch.empty(768,3072,device='cuda');r=torch.empty(1,2,768,device='cuda');s=torch.ones(768,device='cuda')
    with pytest.raises(ValueError,match='dense transposed'):linear(x,w,r,s)
    with pytest.raises(ValueError,match='Unsupported'):linear(x[:1],w,r,s)

import pytest
import torch
import torch.nn.functional as F
from benchmarks.short_ffn import CONFIGS,linear
from fast_moss.ffn import math_library
from fast_moss.normalization import compiler
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from benchmarks.native_layer_norm_tune import bits

RETUNED={(3,1280,5120):('wide',(3,2,20,4,True)),(8,768,3072):('wide',(8,4,12,4,True))}

@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(CONFIGS))
def test_short_ffn_rounding_zero_subnormal_and_graph(monkeypatch,shape):
    strict_precision();torch.manual_seed(6283)
    for s,c in RETUNED.items():monkeypatch.setitem(CONFIGS,s,c)
    m,n,k=shape;mode='gelu' if n>k else 'residual'
    x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')*.02
    r=torch.randn(m,n,device='cuda');s=torch.randn(n,device='cuda');library=math_library();bindings=compiler()
    fn=lambda a,b,c:(linear(a,w,mode,b,c,library,bindings),)
    graph=GraphedCallable(fn,x,r,s)
    for factor in (0.,1.,1e-38,1e20):
        a=x*factor;raw=F.linear(a,w);ref=F.gelu(raw) if mode=='gelu' else r+raw*s
        assert bits(fn(a,r,s)[0],ref) and bits(graph(a,r,s)[0],ref)
    if mode=='residual':
        cancel=-F.linear(x,w)*s;ref=cancel+F.linear(x,w)*s
        assert bits(fn(x,cancel,s)[0],ref) and bits(graph(x,cancel,s)[0],ref)
    positive=torch.full_like(w,.125);minimum=torch.full_like(x,-1.401298464324817e-45)
    zeros=torch.full_like(r,-0.);ones=torch.ones_like(s);raw=F.linear(minimum,positive)
    ref=F.gelu(raw) if mode=='gelu' else zeros+raw*ones
    assert bits(linear(minimum,positive,mode,zeros,ones,library,bindings),ref)


def test_short_ffn_operand_guards():
    with pytest.raises(ValueError,match='contiguous FP32'):linear(torch.randn(4,768),torch.randn(3072,768),'gelu')
    x=torch.empty(4,768,device='cuda');w=torch.empty(3072,768,device='cuda')
    with pytest.raises(ValueError,match='geometry'):linear(x,w,'residual')
    with pytest.raises(ValueError,match='geometry'):linear(x[:1],w,'gelu')
    with pytest.raises(ValueError,match='residual and scale'):linear(torch.empty(4,3072,device='cuda'),w.T.contiguous(),'residual')

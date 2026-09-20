import pytest
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv import linear
from benchmarks.norm_gemv_cache import linear as cached
from benchmarks.native_layer_norm_tune import bits
from fast_moss.loading import strict_precision
from fast_moss.graphs import GraphedCallable

CONFIGS=[(r,v,128 if r==0 else 256,2,4) for r in (0,1) for v in (1,2,4,8)]
CONFIGS += [(0,1,128,2,4,c) for c in ('ca','cg','cs')]+[(1,2,32,2,4,c) for c in ('ca','cg','cs')]

@torch.inference_mode()
@pytest.mark.parametrize('n,mode',[(3840,'none'),(5120,'gelu')])
@pytest.mark.parametrize('config',CONFIGS)
def test_norm_projection_intermediate_rounding_and_graph(n,mode,config):
    strict_precision();torch.manual_seed(168)
    x=torch.randn(1,1280,device='cuda');w=torch.randn(n,1280,device='cuda')*.02
    g=torch.randn(1280,device='cuda');b=torch.randn_like(g)
    helper=cached if len(config)==6 else linear
    def fn(a,gg,bb):return helper(a,w,gg,bb,1e-5,mode,config,debug=True)
    graph=GraphedCallable(fn,x,g,b)
    for a,gg,bb in [(x,g,b),(x*1e-38,g,b),(x*1e10,g,b),(x*.001+10000,g,b),
                    (torch.full_like(x,-0.),g,b),(x,torch.zeros_like(g),torch.full_like(b,-0.)),
                    (x,torch.zeros_like(g),torch.full_like(b,-1.401298464324817e-45))]:
        norm=F.layer_norm(a,(1280,),gg,bb,1e-5);y=F.linear(norm,w);ref=F.gelu(y) if mode=='gelu' else y
        eager=fn(a,gg,bb);captured=graph(a,gg,bb)
        assert bits(eager[0],ref) and bits(captured[0],ref)
        assert bits(eager[1],norm) and bits(captured[1],norm)


def test_norm_gemv_guards_and_dtype():
    with pytest.raises(ValueError,match='aligned'):linear(torch.empty(1,1280),torch.empty(3840,1280),torch.empty(1280),torch.empty(1280),1e-5,'none',(0,1,128,1,4))
    x=torch.zeros(1,1280,device='cuda');w=torch.zeros(3840,1280,device='cuda');g=torch.ones(1280,device='cuda');b=torch.zeros_like(g)
    with pytest.raises(ValueError,match='configuration'):linear(x,w,g,b,1e-5,'none',(0,1,32,1,4))
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert linear(x,w,g,b,1e-5,'none',(0,1,128,1,4)).dtype==torch.float32
    finally:torch.set_default_dtype(old)

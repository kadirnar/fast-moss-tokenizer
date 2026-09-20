import pytest
import torch
import torch.nn.functional as F
from benchmarks.norm_gemv_async import linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision

CONFIGS=[(1,64,64,2,8,4,1),(1,128,128,1,8,4,1),(0,128,256,2,0,4,1),
         (1,32,320,2,8,16,1),(1,256,64,2,8,4,1),(1,64,64,2,8,4,0),
         (1,128,128,1,0,4,0),(1,32,640,2,8,4,1)]

def bits(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('n,mode',[(3840,'none'),(5120,'gelu')])
@pytest.mark.parametrize('cfg',CONFIGS)
def test_async_norm_gemv_exact_shared_lifetime(n,mode,cfg):
    strict_precision();torch.manual_seed(999)
    x=torch.randn(1,1280,device='cuda');w=torch.randn(n,1280,device='cuda')*.03
    g=torch.randn(1280,device='cuda');b=torch.randn_like(g)*.01
    fn=lambda a,ww: (linear(a,ww,g,b,1e-5,mode,cfg),)
    graph=GraphedCallable(fn,x,w)
    for a,ww in [(x,w),(-x,w),(x*1e-38,w),(x*1e10,w),(x*.001+10000,w),
                 (torch.full_like(x,-0.),w),(x,torch.full_like(w,1.401298464324817e-45)),(x,w)]:
        norm=F.layer_norm(a,(1280,),g,b,1e-5);ref=F.linear(norm,ww);ref=F.gelu(ref) if mode=='gelu' else ref
        y,z=linear(a,ww,g,b,1e-5,mode,cfg,debug=True)
        bits(y,ref);bits(z,norm);bits(graph(a,ww)[0],ref)

@torch.inference_mode()
def test_async_norm_gemv_operand_and_shared_memory_guards():
    strict_precision();x=torch.zeros(1,1280,device='cuda');w=torch.zeros(3840,1280,device='cuda');g=torch.ones(1280,device='cuda');b=torch.zeros_like(g)
    cfg=CONFIGS[0]
    with pytest.raises(ValueError,match='aligned'):linear(x,w[:,1:],g,b,1e-5,'none',cfg)
    with pytest.raises(ValueError,match='configuration'):linear(x,w,g,b,1e-5,'none',(1,256,640,2,8,4,1))
    shifted=torch.empty(w.numel()+1,device='cuda')[1:].reshape_as(w)
    with pytest.raises(ValueError,match='aligned'):linear(x,shifted,g,b,1e-5,'none',cfg)

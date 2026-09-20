import pytest
import torch
import torch.nn.functional as F
from benchmarks.residual_gemv_async import linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision

CONFIGS=[(64,128,2,0,4,1,0),(128,320,2,0,4,1,1),(64,320,2,0,4,1,2),
         (32,256,1,16,16,1,2),(64,320,2,16,4,0,2),(128,128,2,16,16,1,0),
         (32,640,2,16,16,1,1)]

def bits(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('k',[1280,5120])
@pytest.mark.parametrize('config',CONFIGS)
def test_residual_staging_bits_and_graph_lifetime(k,config):
    strict_precision();torch.manual_seed(1008)
    x=torch.randn(1,k,device='cuda');w=torch.randn(1280,k,device='cuda')*.03
    r=torch.randn(1,1280,device='cuda');s=torch.randn(1280,device='cuda')*.1
    fn=lambda a,ww,rr,ss:(linear(a,ww,rr,ss,config),)
    graph=GraphedCallable(fn,x,w,r,s)
    for a,ww,rr,ss in [(x,w,r,s),(-x,w,r,s),(x*1e-38,w,torch.zeros_like(r),torch.ones_like(s)),
        (x.abs()*1e-38,w.abs(),torch.zeros_like(r),torch.ones_like(s)),(x*1e10,w,r,s),
        (torch.full_like(x,-0.),w,torch.full_like(r,-0.),torch.full_like(s,-0.)),
        (x,w,-F.linear(x,w)*s,s),(torch.ones_like(x),torch.full_like(w,1.401298464324817e-45),torch.zeros_like(r),torch.ones_like(s)),(x,w,r,s)]:
        ref=rr+F.linear(a,ww)*ss
        for out in (fn(a,ww,rr,ss)[0],graph(a,ww,rr,ss)[0]):bits(out,ref);assert out.stride()==ref.stride()

@torch.inference_mode()
def test_residual_staging_alignment_and_shared_guards():
    strict_precision();x=torch.zeros(1,5120,device='cuda');w=torch.zeros(1280,5120,device='cuda');r=torch.zeros(1,1280,device='cuda');s=torch.ones(1280,device='cuda')
    with pytest.raises(ValueError,match='staging configuration'):linear(x,w,r,s,(256,640,2,16,4,1,1))
    with pytest.raises(ValueError,match='aligned'):linear(x,w[:,1:],r,s,CONFIGS[0])
    shifted=torch.empty(x.numel()+1,device='cuda')[1:].reshape_as(x)
    with pytest.raises(ValueError,match='aligned'):linear(shifted,w,r,s,CONFIGS[0])
    with pytest.raises(ValueError,match='aligned'):linear(x,w,r.reshape(1,1,1280),s,CONFIGS[0])

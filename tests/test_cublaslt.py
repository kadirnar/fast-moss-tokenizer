"""ABI/layout and graph safety checks for the research-only cuBLASLt binding."""
import ctypes
import pytest
import torch
from benchmarks.cublaslt import Algo,Heuristic,LinearPlan,library
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('layout',['col','row','packed'])
def test_pedantic_layout_and_graph_with_tf32_enabled(layout):
    # The plan owns its handle and must ignore, and preserve, PyTorch's math mode.
    old=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=True
    try:
        torch.manual_seed(802)
        w=torch.randn(97,129,device='cuda');x=torch.randn(7,129,device='cuda')
        gold=(x.double()@w.double().T).float()
        with LinearPlan(w,7,layout) as plan:
            assert ctypes.sizeof(Algo)==64 and ctypes.sizeof(Heuristic)==96
            for index in range(len(plan.algorithms)):
                out=plan(x,index)
                torch.testing.assert_close(out,gold,atol=2e-5,rtol=2e-5)
            graph=GraphedCallable(lambda z:(plan(z),),x)
            assert torch.equal(plan(x),graph(x)[0])
            assert torch.equal(plan(-x),graph(-x)[0])
            del graph
        assert torch.backends.cuda.matmul.allow_tf32
        with pytest.raises(ValueError):plan(x)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old


@torch.inference_mode()
def test_shared_workspace_and_plan_validation():
    workspace=torch.empty(32*1024*1024,device='cuda',dtype=torch.uint8)
    w=torch.randn(128,64,device='cuda');x=torch.randn(3,64,device='cuda')
    with LinearPlan(w,3,workspace=workspace) as a,LinearPlan(w,3,'packed',workspace=workspace) as b:
        restored=b.restore(b.metadata(0),library().cublasLtGetVersion())
        assert b.metadata(restored)['opaque']==b.metadata(0)['opaque']
        with pytest.raises(ValueError):b.restore(b.metadata(0),-1)
        graph=GraphedCallable(lambda z:(a(z),b(z,restored)),x)
        y,z=graph(x)
        assert torch.equal(y,a(x)) and torch.equal(z,b(x))
        del graph
        with pytest.raises(ValueError):a(x.double())
        unaligned=torch.empty(x.numel()+1,device='cuda')[1:].view_as(x)
        with pytest.raises(ValueError):a(unaligned)
    with pytest.raises(ValueError):LinearPlan(w,3,workspace=workspace[1:])
    with pytest.raises(ValueError):LinearPlan(w,3,candidates=0)
    with pytest.raises(ValueError):LinearPlan(w.cpu(),3)

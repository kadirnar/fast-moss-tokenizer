import pytest
import torch
import torch.nn.functional as F

from fast_moss.loading import strict_precision
from fast_moss.matrices import MatrixRuntime
from fast_moss.graphs import GraphedCallable
from fast_moss.small_matrices import CONFIGS, linear


@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(CONFIGS))
def test_small_random_weight_rounding(shape):
    strict_precision();torch.manual_seed(723)
    m,n,k=shape
    x=torch.randn(m,k,device='cuda');w=torch.randn(n,k,device='cuda')*.05
    for factor in [0.,1.,-1.,.17,1e-38,1e-20,1e20]:
        z=x*factor
        expected=F.linear(z,w);actual=linear(z,w)
        assert torch.equal(expected.view(torch.int32),actual.view(torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(CONFIGS))
def test_small_runtime_hooks_native_storage_and_capture(shape):
    strict_precision();torch.manual_seed(782)
    m,n,k=shape
    model=torch.nn.Sequential(torch.nn.Linear(k,n,bias=False)).cuda().eval().requires_grad_(False)
    x=torch.randn(1,m,k,device='cuda')
    fallback=torch.randn(m,k,device='cuda').as_strided((m,1,k),(k,1,1))
    expected=model(x);fallback_expected=model(fallback)
    weight=model[0].weight.clone();pointer=model[0].weight.data_ptr()
    observed=[];hook=model[0].register_forward_hook(lambda *args:observed.append(1))
    with MatrixRuntime(model,backend='triton',_profile={'records':[]}) as runtime:
        with pytest.warns(UserWarning,match='CUDA Graph is empty'):
            with pytest.raises(RuntimeError,match='Warm matrix shapes'):
                GraphedCallable(lambda z:(model(z),),x,warmup=0)
        assert torch.equal(model(x),expected)
        assert runtime.small_calls==1 and len(observed)==1
        assert torch.equal(model(fallback),fallback_expected)
        assert runtime.small_calls==1
        graph=GraphedCallable(lambda z:(model(z),),x)
        assert torch.equal(graph(x)[0],expected)
        assert runtime.small_warmed and not runtime.packed and not runtime.plans and not runtime.workspaces
        assert model[0].weight.data_ptr()==pointer and model[0].weight.is_contiguous()
        assert torch.equal(model[0].weight,weight)
    hook.remove()
    assert not runtime.small_warmed and 'forward' not in model[0].__dict__
    assert model[0].weight.data_ptr()==pointer and torch.equal(model(x),expected)
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)


@torch.inference_mode()
def test_small_graph_invalidates_on_later_packing_and_falls_back():
    strict_precision();torch.manual_seed(527)
    model=torch.nn.Sequential(torch.nn.Linear(3072,768,bias=False)).cuda().eval().requires_grad_(False)
    small=torch.randn(12,3072,device='cuda');large=torch.randn(24,3072,device='cuda')
    small_ref=model(small);large_ref=model(large)
    with MatrixRuntime(model,backend='triton',_profile={'records':[]}) as runtime:
        graph=GraphedCallable(lambda z:(model(z),),small)
        calls=runtime.small_calls
        assert torch.equal(graph(small)[0],small_ref)
        assert torch.equal(model(large),large_ref)
        assert not model[0].weight.is_contiguous()
        with pytest.raises(RuntimeError,match='storage changed'):graph(small)
        assert torch.equal(model(small),small_ref)
        assert runtime.small_calls==calls
    assert model[0].weight.is_contiguous() and torch.equal(model(small),small_ref)


def test_small_helper_validates_operands():
    with pytest.raises(ValueError,match='supported contiguous'):
        linear(torch.randn(3,1280),torch.randn(5120,1280))
    with pytest.raises(ValueError,match='supported contiguous'):
        linear(torch.randn(1280),torch.randn(5120,1280))


@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(s for s in CONFIGS if s[0]==1))
def test_gemv_vector_rank_and_signed_zero(shape):
    strict_precision();torch.manual_seed(158)
    _,n,k=shape
    model=torch.nn.Sequential(torch.nn.Linear(k,n,bias=False)).cuda().eval().requires_grad_(False)
    for underflow in [False,True]:
        if underflow:
            model[0].weight.fill_(.125)
            vector=torch.full((k,),-1.401298464324817e-45,device='cuda')
        else:
            vector=torch.randn(k,device='cuda')
        inputs=[vector,vector.reshape(1,k),vector.reshape(1,1,k)]
        expected=[model(x) for x in inputs]
        if underflow:
            assert all(torch.count_nonzero(y.view(torch.int32))==0 for y in expected)
        with MatrixRuntime(model,backend='triton',_profile={'records':[]}) as runtime:
            for x,ref in zip(inputs,expected):
                actual=model(x)
                assert torch.equal(ref.view(torch.int32),actual.view(torch.int32))
                graph=GraphedCallable(lambda z:(model(z),),x)
                assert torch.equal(ref.view(torch.int32),graph(x)[0].view(torch.int32))
                del graph
            assert runtime.small_calls>0

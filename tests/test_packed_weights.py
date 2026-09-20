import pytest
import torch
from benchmarks.cublaslt import LinearPlan,library
from benchmarks.cublaslt_model import experimental
from benchmarks.packed_weights import PackedWeights
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('resident',[False,True])
def test_shared_packing_graph_fallback_and_restoration(resident):
    torch.manual_seed(382)
    model=torch.nn.Sequential(torch.nn.Linear(64,128,bias=False)).cuda().eval().requires_grad_(False)
    # Integer products make the vendor ordering irrelevant for this lifetime test.
    model[0].weight.copy_(torch.randint(-3,4,model[0].weight.shape,device='cuda'))
    original=model[0].weight.clone();identity=id(model[0].weight)
    selected={}
    inputs=[torch.randint(-3,4,(m,64),device='cuda').float() for m in [3,7]]
    for x in inputs:
        with LinearPlan(model[0].weight,x.shape[0],'packed') as plan:
            selected[(x.shape[0],128,64)]={'backend':'lt_packed_0','algorithm':plan.metadata(0),
                                         'cublaslt_version':library().cublasLtGetVersion()}
    refs=[model(x) for x in inputs]
    fallback=torch.randn(5,64,device='cuda');fallback_ref=model(fallback)
    with experimental(model,selected,resident=resident) as plans:
        with pytest.raises(RuntimeError):
            with experimental(model,selected):pass
        for x,ref in zip(inputs,refs):
            assert torch.equal(model(x),ref)
            graph=GraphedCallable(lambda z:(model(z),),x)
            assert torch.equal(graph(x)[0],ref)
            del graph
        assert len(plans)==2
        assert len({p.weight.data_ptr() for p,_ in plans.values()})==1
        assert torch.equal(model[0].weight,original)
        assert id(model[0].weight)==identity
        assert model[0].weight.is_contiguous()==(not resident)
        graph=GraphedCallable(lambda z:(model(z),),fallback)
        assert torch.equal(graph(fallback)[0],fallback_ref)
        del graph
        with torch.autocast('cuda',dtype=torch.bfloat16):
            assert model(inputs[0]).dtype==torch.bfloat16
    assert plans=={}
    assert model[0].weight.is_contiguous()
    assert torch.equal(model[0].weight,original)
    assert torch.equal(model(fallback),fallback_ref)
    assert id(model[0].weight)==identity
    assert '_fast_cublaslt_active' not in model.__dict__


@torch.inference_mode()
def test_resident_exception_restores_values_and_rejects_storage_aliases():
    model=torch.nn.Sequential(torch.nn.Linear(64,128,bias=False)).cuda().eval().requires_grad_(False)
    ref=model[0].weight.clone()
    packing=PackedWeights(model,resident=True)
    try:
        packing.get(model[0].weight)
        assert not model[0].weight.is_contiguous()
    finally:
        packing.close()
    assert torch.equal(model[0].weight,ref) and model[0].weight.is_contiguous()
    model.register_parameter('alias',torch.nn.Parameter(model[0].weight.view_as(model[0].weight),requires_grad=False))
    packing=PackedWeights(model,resident=True)
    with pytest.raises(ValueError,match='aliased'):packing.get(model[0].weight)
    assert model[0].weight.is_contiguous()


@torch.inference_mode()
def test_failed_plan_restores_resident_storage(monkeypatch):
    model=torch.nn.Sequential(torch.nn.Linear(64,128,bias=False)).cuda().eval().requires_grad_(False)
    original=model[0].weight.clone()
    def fail(*args):raise RuntimeError('injected algorithm validation failure')
    monkeypatch.setattr(LinearPlan,'restore',fail)
    selected={(3,128,64):{'backend':'lt_packed_0','algorithm':{},'cublaslt_version':0}}
    with pytest.raises(RuntimeError,match='injected'):
        with experimental(model,selected,resident=True):
            model(torch.zeros(3,64,device='cuda'))
    assert torch.equal(model[0].weight,original)
    assert model[0].weight.is_contiguous()
    assert 'forward' not in model[0].__dict__
    assert '_fast_cublaslt_active' not in model.__dict__


@torch.inference_mode()
@pytest.mark.parametrize('resident',[False,True])
def test_singleton_stride_preserves_batched_matmul_dispatch(resident):
    from benchmarks.cublaslt_model import folds_to_mm
    torch.manual_seed(937)
    model=torch.nn.Sequential(torch.nn.Linear(1280,768,bias=False)).cuda().eval().requires_grad_(False)
    flat=torch.randn(128,1280,device='cuda')
    x=flat.as_strided((128,1,1280),(1280,1,1))
    assert x.is_contiguous() and not folds_to_mm(x) and folds_to_mm(flat[:,None])
    ref=model(x)
    selected={(128,768,1280):{'backend':'lt_packed_0','algorithm':{},'cublaslt_version':0}}
    # A bad descriptor proves this layout must fall back before constructing a plan.
    with experimental(model,selected,resident=resident) as plans:
        assert torch.equal(model(x),ref)
        graph=GraphedCallable(lambda z:(model(z),),x)
        assert torch.equal(graph(x)[0],ref)
        assert plans=={}
        del graph
    # The batched path must also stay exact after another shape has packed this weight.
    with LinearPlan(model[0].weight,128,'packed') as plan:
        selected[(128,768,1280)]={'backend':'lt_packed_0','algorithm':plan.metadata(0),
                                'cublaslt_version':library().cublasLtGetVersion()}
    with experimental(model,selected,resident=resident) as plans:
        model(flat)
        assert len(plans)==1
        assert torch.equal(model(x),ref)
        graph=GraphedCallable(lambda z:(model(z),),x)
        assert torch.equal(graph(x)[0],ref)
        del graph

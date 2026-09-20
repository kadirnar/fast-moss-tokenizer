from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
import torch

from benchmarks.fixtures import structural_model
from benchmarks.quantizer_prepare import reference
from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized
from fast_moss.quantizer import decode_latents
from fast_moss.quantizer_prepare import SOURCE, compile_kernel, prepare


OPTIONS = dict(norm_backend='cuda', quantizer_backend='triton')


def fixture(monkeypatch):
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    model = structural_model()
    return model, model.quantizer.quantizers[0]


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.view(torch.int32) if a.dtype == torch.float32 else a,
                       b.view(torch.int32) if b.dtype == torch.float32 else b)


@torch.inference_mode()
def test_prepare_supported_kernel_matches_research_binary_and_intermediates():
    from benchmarks.quantizer_prepare import SOURCE as research_source
    assert SOURCE == research_source
    torch.manual_seed(1560)
    for batch, time in [(1,1),(1,3),(1,129),(8,3),(128,3),(257,17)]:
        x = torch.randn(batch,8,time,device='cuda')
        graph = GraphedCallable(prepare, x)
        for value in [x, x*1.e-38, x*1.e-20, x*1.e18, x*1.e20,
                      torch.full_like(x,-0.), torch.full_like(x,float('nan'))]:
            refs = reference(value)
            for out in [prepare(value),graph(value)]:
                for a,b in zip(out,refs):
                    bits(a,b)
                    assert a.stride() == b.stride()
        del graph
    prior = json.loads(Path('results/quantizer_prepare_probe.json').read_text())
    assert compile_kernel()[2] == prior['resources']


@torch.inference_mode()
@pytest.mark.parametrize('shape', [(1,8,1),(1,8,3),(8,8,3),(1,8,40)])
@pytest.mark.parametrize('ste', [False,True])
def test_owned_quantizer_switch_graph_replay_and_restore(monkeypatch, shape, ste):
    model,q = fixture(monkeypatch)
    original = q.decode_latents
    original_forward = q.forward
    x = torch.randn(shape,device='cuda')*.2
    def ref(z):
        value,index = original(z)
        return (z+(value-z) if ste else value),index
    pointers = [p.data_ptr() for p in model.parameters()]
    with optimized(model, **OPTIONS):
        owner = model._fast_norm_runtime
        fn = lambda z: decode_latents(q,z,straight_through=ste)
        staged = GraphedCallable(fn,x)
        before = owner.quantizer_prepare_calls
        owner.quantizer_prepare_enabled = False
        direct = GraphedCallable(fn,x)
        assert owner.quantizer_prepare_calls == before
        for z in [x,-x,x*1.e-20,x*1.e10,x]:
            expected = ref(z)
            for out in [staged(z),direct(z),fn(z)]:
                for a,b in zip(out,expected):
                    bits(a,b)
                    assert a.stride() == b.stride()
        owner.quantizer_prepare_enabled = True
        fn(x)
        assert owner.quantizer_prepare_calls == before+1
        assert [p.data_ptr() for p in model.parameters()] == pointers
    assert q.decode_latents == original and q.forward == original_forward
    assert not hasattr(q,'_fast_quantizer_prepare_runtime')
    assert not owner.quantizer_forwards and not owner.quantizer_prepare_warmed
    for graph in [staged,direct]:
        with pytest.raises(RuntimeError,match='storage changed'):
            graph(x)
    with pytest.raises(RuntimeError,match='active owner'):
        owner.prepare_quantizer(q,x)


def test_prepare_layout_grad_autocast_and_owner_fallbacks(monkeypatch):
    model,q = fixture(monkeypatch)
    x = torch.randn(2,8,3,device='cuda')
    with optimized(model,**OPTIONS):
        owner = model._fast_norm_runtime
        invalid = [x.cpu(),x.half(),x[...,::2],x[:,::2],x[:0],x[...,:0],x.reshape(2,24)]
        for value in invalid:
            assert owner.prepare_quantizer(q,value) is None
        assert owner.prepare_quantizer(q,x.detach().requires_grad_(True)) is None
        with torch.autocast('cuda'):
            assert owner.prepare_quantizer(q,x) is None
        assert owner.quantizer_prepare_calls == 0
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError,match='owner thread'):
                pool.submit(owner.prepare_quantizer,q,x).result()
        assert owner.prepare_quantizer(q,x) is not None


@torch.inference_mode()
def test_prepare_warmup_per_module_shape_stream_and_failed_launch(monkeypatch):
    model,q = fixture(monkeypatch)
    x = torch.randn(1,8,1,device='cuda')
    with optimized(model,**OPTIONS):
        owner = model._fast_norm_runtime
        owner.quantizer_prepare_enabled = False
        assert owner.prepare_quantizer(q,x) is None
        owner.quantizer_prepare_enabled = True
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm quantizer'):
                owner.prepare_quantizer(q,x)
        import fast_moss.quantizer_prepare as helper
        def fail(x): raise RuntimeError('launch failed')
        with monkeypatch.context() as patch:
            patch.setattr(helper,'prepare',fail)
            with pytest.raises(RuntimeError,match='launch failed'):
                owner.prepare_quantizer(q,x)
        assert not owner.quantizer_prepare_warmed and not owner.quantizer_prepare_calls
        assert owner.prepare_quantizer(q,x) is not None
        for module,value in [(model.quantizer.quantizers[1],x),(q,torch.randn(1,8,3,device='cuda'))]:
            with monkeypatch.context() as patch:
                patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
                with pytest.raises(RuntimeError,match='Warm quantizer'):
                    owner.prepare_quantizer(module,value)
            assert owner.prepare_quantizer(module,value) is not None
        parent = torch.cuda.current_stream()
        side = torch.cuda.Stream();side.wait_stream(parent)
        with torch.cuda.stream(side):
            with monkeypatch.context() as patch:
                patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
                with pytest.raises(RuntimeError,match='Warm quantizer'):
                    owner.prepare_quantizer(q,x)
            assert owner.prepare_quantizer(q,x) is not None
        parent.wait_stream(side)
        assert len(owner.quantizer_prepare_warmed) == 4


@torch.inference_mode()
@pytest.mark.parametrize('method',['forward','decode_latents'])
def test_prepare_changed_methods_and_preexisting_overrides(monkeypatch,method):
    model,q = fixture(monkeypatch)
    x = torch.randn(1,8,3,device='cuda')
    with optimized(model,**OPTIONS):
        owner = model._fast_norm_runtime
        saved = getattr(q,method)
        setattr(q,method,lambda *a,**kw:saved(*a,**kw))
        assert owner.prepare_quantizer(q,x) is None
        setattr(q,method,saved)
        assert owner.prepare_quantizer(q,x) is not None
    previous = getattr(q,method)
    override = lambda *a,**kw:previous(*a,**kw)
    setattr(q,method,override)
    with optimized(model,**OPTIONS):
        assert q not in model._fast_norm_runtime.quantizer_forwards
        assert not hasattr(q,'_fast_quantizer_prepare_runtime')
    assert getattr(q,method) is override


@torch.inference_mode()
def test_prepare_observers_public_quantizer_and_exception_cleanup(monkeypatch):
    model,q = fixture(monkeypatch)
    x = torch.randn(1,1,5760,device='cuda')*.05
    captured = []
    hook = model.quantizer.register_forward_pre_hook(lambda m,a:captured.append(a))
    ref = model._encode_frame(x)
    hook.remove()
    z,n,_ = captured[0]
    qref = model.quantizer(z,n)
    original = q.decode_latents
    observed = []
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,**OPTIONS):
            owner = model._fast_norm_runtime
            hook = model.quantizer.register_forward_hook(lambda m,a,o:observed.append(o))
            try:
                out = model._encode_frame(x)
            finally:
                hook.remove()
            for key in ref: bits(ref[key],out[key])
            for a,b in zip(qref,observed[0]): bits(a,b)
            assert owner.quantizer_prepare_calls == 32
            raise RuntimeError('body failure')
    assert q.decode_latents == original
    assert not owner.quantizer_forwards and not owner.quantizer_prepare_warmed
    assert not hasattr(q,'_fast_quantizer_prepare_runtime')
    assert not hasattr(model,'_fast_norm_runtime')


@torch.inference_mode()
def test_prepare_requires_both_selected_backends(monkeypatch):
    model,q = fixture(monkeypatch)
    for opts in [dict(norm_backend='cuda'),dict(quantizer_backend='triton')]:
        with optimized(model,**opts):
            assert not hasattr(q,'_fast_quantizer_prepare_runtime')

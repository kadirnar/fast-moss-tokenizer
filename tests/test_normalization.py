from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from fast_moss.graphs import GraphedCallable
from fast_moss.normalization import CONFIGS, NormalizationRuntime, compiler, owned_forward
from fast_moss.optimize import optimized
from tests.test_ffn import fixture


def exact(a, b):
    assert torch.equal(a.view(torch.int32), b.view(torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize('shape', sorted(CONFIGS))
def test_profiled_shapes_values_graph_and_restore(monkeypatch, shape):
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    rows, n = shape
    layer = torch.nn.LayerNorm(n).cuda().eval().requires_grad_(False)
    model = torch.nn.Sequential(layer).eval()
    torch.manual_seed(124)
    layer.weight.copy_(torch.randn_like(layer.weight));layer.bias.copy_(torch.randn_like(layer.bias))
    x = torch.randn(1, rows, n, device='cuda')
    values = [x, -x, x*1e-38, x*1e20, x*0+1e4, torch.full_like(x, -0.),
              torch.full_like(x, float("inf")), torch.full_like(x, float("nan"))]
    refs = [layer(z) for z in values]
    original = layer.forward
    with optimized(model, norm_backend='cuda'):
        runtime = model._fast_norm_runtime
        saved = layer.forward
        assert owned_forward(layer)
        graph = GraphedCallable(lambda z: (layer(z),), x)
        for z, ref in zip(values, refs):
            exact(layer(z), ref);exact(graph(z)[0], ref)
        assert runtime.calls > 0
    assert layer.forward == original and not hasattr(layer, '_fast_norm_runtime')
    with pytest.raises(RuntimeError, match='storage changed'):graph(x)
    with pytest.raises(RuntimeError, match='active owner'):saved(x)


@torch.inference_mode()
@pytest.mark.parametrize('backend', ['triton', 'cute'])
@pytest.mark.parametrize('encoder', [False, True])
def test_ffn_norm_coexistence_hooks_custom_forward_and_restoration(monkeypatch, backend, encoder):
    model, layer, _ = fixture(monkeypatch, encoder=encoder)
    x = torch.randn(1, 1, 1280, device='cuda');ref = layer._ff_block(x)
    opts = dict(matrix_backend='triton', ffn_backend='triton', residual_backend=backend, norm_backend='cuda')
    with optimized(model, **opts):
        runtime = model._fast_matrix_runtime;norm = model._fast_norm_runtime
        exact(layer._ff_block(x), ref)
        assert runtime.ffn_gemv_calls == 2 and norm.calls == 1
        graph = GraphedCallable(lambda z: (layer._ff_block(z),), x)
        exact(graph(x)[0], ref)
        before = runtime.ffn_gemv_calls
        seen = []
        hook = layer.norm2.register_forward_hook(lambda m, a, o: seen.append(o.clone()))
        try:
            exact(layer._ff_block(x), ref)
            assert len(seen) == 1 and runtime.ffn_gemv_calls == before
        finally:hook.remove()
        forward = layer.norm2.forward
        layer.norm2.forward = lambda z: forward(z)+.01
        assert not owned_forward(layer.norm2)
        layer._ff_block(x)
        assert runtime.ffn_gemv_calls == before
    assert 'forward' not in layer.norm2.__dict__
    custom = layer.norm2.forward
    layer.norm2.forward = lambda z: custom(z)+.01
    expected = layer._ff_block(x);method = layer.norm2.forward
    with optimized(model, **opts):
        exact(layer._ff_block(x), expected)
        assert model._fast_norm_runtime.calls == model._fast_matrix_runtime.ffn_gemv_calls == 0
    assert layer.norm2.forward is method


def test_fallback_grad_layout_epsilon_and_owner_guards(monkeypatch):
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    layer = torch.nn.LayerNorm(768).cuda().eval().requires_grad_(False)
    model = torch.nn.Sequential(layer).eval()
    original = layer.forward
    x = torch.randn(2, 768, device='cuda')
    inputs = [x.T.contiguous().T, x[:1], torch.randn(1537, device='cuda')[1:].reshape(2,768)]
    with optimized(model, norm_backend='cuda'):
        runtime = model._fast_norm_runtime
        for z in inputs:exact(layer(z), original(z))
        assert runtime.calls == 0
        grad = x.detach().requires_grad_(True)
        actual = torch.autograd.grad(layer(grad).square().sum(), grad)[0]
        expected = torch.autograd.grad(original(grad).square().sum(), grad)[0]
        exact(actual, expected)
        layer.eps = 0.;exact(layer(x), original(x));layer.eps = 1e-5
        with torch.autocast('cuda'):exact(layer(x), original(x))
        assert runtime.calls == 0
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError, match='owner thread'):pool.submit(layer, x).result()
        stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            exact(layer(x), original(x))
            graph = GraphedCallable(lambda z: (layer(z),), x)
            exact(graph(x)[0], original(x))
        torch.cuda.current_stream().wait_stream(stream)
        assert runtime.calls > 0
    assert layer.forward == original


def test_options_compiler_and_exception_cleanup(monkeypatch):
    model = torch.nn.Linear(1,1).eval().requires_grad_(False)
    with pytest.raises(ValueError, match='normalization'):
        with optimized(model, norm_backend='bad'):pass
    monkeypatch.setattr('fast_moss.normalization.version', lambda name: 'unvalidated')
    with pytest.raises(ValueError, match='normalization extra'):compiler()
    monkeypatch.undo()
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    model = torch.nn.Sequential(torch.nn.LayerNorm(768)).cuda().eval().requires_grad_(False)
    with pytest.raises(RuntimeError, match='body failure'):
        with optimized(model, norm_backend='cuda'):
            raise RuntimeError('body failure')
    assert 'forward' not in model[0].__dict__ and not hasattr(model, '_fast_norm_runtime')
    runtime = NormalizationRuntime(model)
    with runtime:pass
    with pytest.raises(RuntimeError, match='has been used'):
        with runtime:pass


@torch.inference_mode()
def test_entry_invalidates_graphs_and_capture_requires_warmup(monkeypatch):
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    model = torch.nn.Sequential(torch.nn.LayerNorm(768)).cuda().eval().requires_grad_(False)
    x = torch.randn(2, 768, device='cuda')
    graph = GraphedCallable(lambda z: (model(z),), x)
    with optimized(model, norm_backend='cuda'):
        with pytest.raises(RuntimeError, match='storage changed'):graph(x)
        # Exercise the pre-capture guard without leaving CUDA in a failed capture.
        monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
        with pytest.raises(RuntimeError, match='Warm CUDA LayerNorm'):model(x)
        monkeypatch.undo()
    assert not hasattr(model, '_fast_norm_runtime')


def test_streaming_owner_and_trainable_model_rejected(monkeypatch):
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: {})
    model = torch.nn.Sequential(torch.nn.LayerNorm(768)).cuda().eval().requires_grad_(False)
    model._fast_streaming_owner = object()
    with pytest.raises(RuntimeError, match='streaming session'):
        with optimized(model, norm_backend='cuda'):pass
    assert not hasattr(model, '_fast_norm_runtime') and 'forward' not in model[0].__dict__
    del model._fast_streaming_owner
    runtime = NormalizationRuntime(model)
    model[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen FP32'):
        with runtime:pass
    assert not hasattr(model, '_fast_norm_runtime')

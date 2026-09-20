from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from fast_moss.cublaslt import LinearPlan, library
from fast_moss.graphs import GraphedCallable
from fast_moss.matrices import MatrixRuntime, load_profile
from fast_moss.optimize import optimized
from benchmarks.fixtures import structural_model


def fixture():
    torch.manual_seed(438)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False)).cuda().eval().requires_grad_(False)
    with torch.no_grad():
        model[0].weight.copy_(torch.randint(-3, 4, model[0].weight.shape, device='cuda'))
    records = []
    for m in [3, 7]:
        with LinearPlan(model[0].weight, m, 'packed') as plan:
            records.append({'shape': [m, 128, 64], 'algorithm': plan.metadata(0)})
    profile = {'cublaslt_version': library().cublasLtGetVersion(), 'records': records}
    x = torch.randint(-3, 4, (3, 64), device='cuda').float()
    return model, profile, x


@torch.inference_mode()
@pytest.mark.parametrize('backend', ['cublaslt', 'triton'])
def test_runtime_storage_graph_epochs_stream_workspaces_and_restoration(backend):
    model, profile, x = fixture()
    expected, weights = model(x), model[0].weight.clone()
    identity = id(model[0].weight)
    old = GraphedCallable(lambda y: (model(y),), x)
    with MatrixRuntime(model, backend=backend, _profile=profile) as runtime:
        assert runtime.packed_bytes == 0 and model[0].weight.is_contiguous()
        assert torch.equal(model(x), expected)
        assert runtime.packed_bytes == weights.numel() * 4
        assert torch.equal(model[0].weight, weights) and not model[0].weight.is_contiguous()
        with pytest.raises(RuntimeError, match='storage changed'):
            old(x)
        assert torch.equal(model(x), expected)
        graph = GraphedCallable(lambda y: (model(y),), x)
        assert torch.equal(graph(x)[0], expected)
        assert len(runtime.workspaces) == 2
        assert len({w.data_ptr() for w in runtime.workspaces.values()}) == 2
        assert len(runtime.plans) == 2
        with pytest.raises(RuntimeError):
            with MatrixRuntime(model, _profile=profile):
                pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match='host thread'):
                pool.submit(model, x).result()
    assert id(model[0].weight) == identity
    assert torch.equal(model[0].weight, weights) and model[0].weight.is_contiguous()
    assert torch.equal(model(x), expected)
    assert 'forward' not in model[0].__dict__
    assert not runtime.plans and not runtime.workspaces and not runtime.packed
    assert '_fast_matrix_runtime' not in model.__dict__
    with pytest.raises(RuntimeError, match='storage changed'):
        graph(x)
    with pytest.raises(RuntimeError):
        runtime.__enter__()
    current = GraphedCallable(lambda y: (model(y),), x)
    assert torch.equal(current(x)[0], expected)


@torch.inference_mode()
def test_runtime_preserves_fallback_dispatch_hooks_and_custom_forward():
    model, profile, x = fixture()
    inputs = [torch.randn(5, 64, device='cuda'), torch.randn(3, 128, device='cuda')[:, ::2],
              torch.randn(3 * 64 + 1, device='cuda')[1:].view(3, 64),
              torch.randn(3, 64, device='cuda').as_strided((3, 1, 64), (64, 1, 1))]
    refs = [model(y) for y in inputs]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        autocast_ref = model(x)
    calls = []
    hook = model[0].register_forward_hook(lambda *args: calls.append(1))
    with MatrixRuntime(model, _profile=profile) as runtime:
        for value, ref in zip(inputs, refs):
            assert torch.equal(model(value), ref)
        assert not runtime.plans and len(calls) == len(inputs)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            assert torch.equal(model(x), autocast_ref)
    hook.remove()
    original = model[0].forward
    model[0].forward = lambda value: original(value) + 2
    expected = model(x)
    with MatrixRuntime(model, _profile=profile) as runtime:
        assert not runtime.packed and torch.equal(model(x), expected)
    assert 'forward' in model[0].__dict__


@torch.inference_mode()
@pytest.mark.parametrize('kind', ['parameter', 'buffer', 'tied'])
def test_runtime_rejects_registered_storage_aliases_before_mutation(kind):
    model, profile, x = fixture()
    ptr = model[0].weight.data_ptr()
    if kind == 'parameter':
        model.register_parameter('alias', torch.nn.Parameter(model[0].weight.view_as(model[0].weight), requires_grad=False))
    elif kind == 'buffer':
        model.register_buffer('alias', model[0].weight.detach())
    else:
        model.register_parameter('alias', model[0].weight)
    with pytest.raises(ValueError, match='unaliased'):
        with MatrixRuntime(model, _profile=profile):
            pass
    assert model[0].weight.data_ptr() == ptr and model[0].weight.is_contiguous()
    assert '_fast_matrix_runtime' not in model.__dict__


@torch.inference_mode()
def test_runtime_failed_plan_restores_and_profile_failure_unwinds_optimizer(monkeypatch):
    model, profile, x = fixture()
    expected = model(x)
    def fail(*args):
        raise RuntimeError('injected restore failure')
    monkeypatch.setattr(LinearPlan, 'restore', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with MatrixRuntime(model, _profile=profile):
            model(x)
    assert model[0].weight.is_contiguous() and torch.equal(model(x), expected)
    assert 'forward' not in model[0].__dict__
    fixture_model = structural_model()
    with pytest.raises(ValueError, match='pinned'):
        with optimized(fixture_model, matrix_backend='cublaslt'):
            pass
    assert '_fast_optimization_active' not in fixture_model.__dict__
    with pytest.raises(ValueError, match='Unknown matrix'):
        with optimized(model, matrix_backend='bad'):
            pass
    monkeypatch.setattr('fast_moss.matrices.load_profile', lambda model: profile)
    with pytest.raises(RuntimeError, match='injected'):
        with optimized(model, matrix_backend='cublaslt'):
            model(x)
    assert '_fast_optimization_active' not in model.__dict__
    assert model[0].weight.is_contiguous() and torch.equal(model(x), expected)


@torch.inference_mode()
def test_runtime_rejects_open_session_tf32_and_context_inside_graph():
    model, profile, x = fixture()
    model._fast_streaming_owner = object()
    with pytest.raises(RuntimeError, match='streaming session'):
        with MatrixRuntime(model, _profile=profile):
            pass
    del model._fast_streaming_owner
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with pytest.raises(ValueError, match='TF32'):
            with MatrixRuntime(model, _profile=profile):
                pass
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    def nested(value):
        with MatrixRuntime(model, _profile=profile):
            return (model(value),)
    with pytest.raises(RuntimeError, match='enclose graph construction'):
        GraphedCallable(nested, x)
    assert model[0].weight.is_contiguous()


@torch.inference_mode()
def test_runtime_requires_capture_stream_warmup():
    model, profile, x = fixture()
    with MatrixRuntime(model, _profile=profile):
        # No model call has populated plans on the graph's side stream.
        with pytest.warns(UserWarning, match='CUDA Graph is empty'):
            with pytest.raises(RuntimeError, match='Warm matrix shapes'):
                GraphedCallable(lambda y: (model(y),), x, warmup=0)


@torch.inference_mode()
def test_runtime_rejects_unmatched_library_profile(monkeypatch):
    model = structural_model()
    monkeypatch.setattr(torch, '__version__', 'unvalidated-version')
    with pytest.raises(ValueError, match='recorded GPU'):
        load_profile(model)


@torch.inference_mode()
def test_lazy_packing_invalidates_graphs_of_previously_unpacked_weights():
    model, profile, x = fixture()
    fallback = torch.randn(5, 64, device='cuda')
    ref = model(fallback)
    with MatrixRuntime(model, _profile=profile) as runtime:
        early = GraphedCallable(lambda y: (model(y),), fallback)
        assert not runtime.packed and torch.equal(early(fallback)[0], ref)
        model(x)
        with pytest.raises(RuntimeError, match='storage changed'):
            early(fallback)
        late = GraphedCallable(lambda y: (model(y),), fallback)
        assert torch.equal(late(fallback)[0], ref)
        model(torch.ones(7, 64, device='cuda'))  # New plan, same packed storage.
        assert torch.equal(late(fallback)[0], ref)


@torch.inference_mode()
@pytest.mark.parametrize('shape', [(5120,1280), (1280,5120)])
def test_ordered_runtime_arithmetic_hooks_fallback_and_storage(shape):
    import json
    from pathlib import Path
    from fast_moss.loading import strict_precision
    strict_precision()
    torch.manual_seed(954)
    n, k = shape
    model = torch.nn.Sequential(torch.nn.Linear(k, n, bias=False)).cuda().eval().requires_grad_(False)
    x = torch.randn(8,3,k,device='cuda')
    fallback = torch.randn(8,1,k,device='cuda').as_strided((8,1,k),(k,1,1))
    ref, fallback_ref = model(x), model(fallback)
    weight = model[0].weight.clone()
    profile = json.loads(Path('fast_moss/matrix_profile.json').read_text())
    observed = []
    hook = model[0].register_forward_hook(lambda *args: observed.append(1))
    with MatrixRuntime(model, backend='triton', _profile=profile) as runtime:
        assert torch.equal(model(x), ref)
        assert runtime.triton_calls == 1 and len(observed) == 1
        assert torch.equal(model(fallback), fallback_ref)
        assert runtime.triton_calls == 1
        graph = GraphedCallable(lambda z:(model(z),), x)
        assert torch.equal(graph(x)[0], ref)
        assert torch.equal(model[0].weight, weight)
    hook.remove()
    assert model[0].weight.is_contiguous() and torch.equal(model[0].weight, weight)
    assert torch.equal(model(x), ref)
    with pytest.raises(RuntimeError, match='storage changed'):
        graph(x)


def test_ordered_runtime_compiler_gate_and_helper_validation(monkeypatch):
    import triton
    from fast_moss.ordered_matrices import linear
    with pytest.raises(ValueError, match='supported contiguous'):
        linear(torch.randn(24,1280), torch.randn(1280,5120))
    monkeypatch.setattr(triton, '__version__', 'unvalidated')
    with pytest.raises(ValueError, match='validated Triton'):
        MatrixRuntime(torch.nn.Linear(1,1), backend='triton')

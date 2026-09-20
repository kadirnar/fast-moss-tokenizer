import pytest
import torch
import torch.nn.functional as F
from fast_moss.quantizer import select
from fast_moss.graphs import GraphedCallable


def reference(dots, norm, cnorm, book, latents, ste):
    b, d, t = latents.shape
    indices = (-(norm - dots + cnorm)).max(1).indices.reshape(b, t)
    values = F.embedding(indices, book).transpose(1, 2)
    return (latents + (values - latents) if ste else values), indices


@torch.inference_mode()
@pytest.mark.parametrize('shape', [(1, 8, 1, 1024), (8, 8, 3, 1024), (2, 7, 37, 1023), (3, 8, 128, 1024)])
@pytest.mark.parametrize('ste', [False, True])
@pytest.mark.parametrize('gapped', [False, True])
def test_quantizer_postprocess_exact_graph_strides_and_ties(shape, ste, gapped):
    torch.manual_seed(917)
    b, d, t, k = shape
    dots = torch.randn(b * t, k, device='cuda')
    norm = torch.randn(b * t, 1, device='cuda')
    cnorm = torch.randn(1, k, device='cuda')
    book = torch.randn(k, d, device='cuda')
    # Noncontiguous latents exercise the fused straight-through load strides.
    latents = torch.randn(b, d, t * 2, device='cuda')[..., ::2] if gapped else torch.randn(b, d, t, device='cuda')
    dots[0, 7] = dots[0, 31] = 100
    cnorm[0, 7] = cnorm[0, 31] = 0
    expected = reference(dots, norm, cnorm, book, latents, ste)
    fn = lambda x: select(x, norm, cnorm, book, latents, straight_through=ste)
    actual = fn(dots)
    assert actual[1][0, 0].item() == 7
    for x, y in zip(expected, actual): assert torch.equal(x, y)
    assert actual[0].stride() == expected[0].stride()
    graph = GraphedCallable(fn, dots)
    for x, y in zip(expected, graph(dots)): assert torch.equal(x, y)


@torch.inference_mode()
@pytest.mark.parametrize('ste', [False, True])
def test_quantizer_rounding_boundaries_nan_and_infinity(ste):
    # Dropping the common norm would choose 1, but FP32 subtraction rounds both
    # distances to one value, and upstream selects the first code instead.
    dots = torch.zeros(5, 1024, device='cuda')
    dots[0, 1] = 1
    dots[1, 71] = dots[1, 503] = float('nan')
    dots[2, 18] = dots[2, 32] = float('inf')
    dots[3] = -float('inf')
    dots[4, 32] = -float('inf'); dots[4, 913] = float('nan')
    norm = torch.tensor([2**26, 1, 1, 1, 1], device='cuda', dtype=torch.float32)[:, None]
    cnorm = torch.ones(1, 1024, device='cuda')
    book = torch.randn(1024, 8, device='cuda')
    latents = torch.randn(1, 8, 5, device='cuda') * 1e10
    expected = reference(dots, norm, cnorm, book, latents, ste)
    actual = select(dots, norm, cnorm, book, latents, straight_through=ste)
    assert expected[1].tolist() == [[0, 71, 18, 0, 913]]
    for x, y in zip(expected, actual): assert torch.equal(x, y)


def test_quantizer_rejects_invalid_storage():
    with pytest.raises(ValueError):
        select(torch.zeros(1, 1024), torch.zeros(1, 1), torch.zeros(1, 1024),
               torch.zeros(1024, 8), torch.zeros(1, 8, 1))


@torch.inference_mode()
def test_full_quantizer_masked_outputs_counts_and_encoder_hooks():
    from benchmarks.fixtures import structural_model
    from fast_moss.optimize import optimized
    model = structural_model()
    x = torch.randn(3, 1, 3840, device='cuda') * .05
    lengths = torch.tensor([3840, 1920, 0], device='cuda')
    original_encode = model._encode_frame
    ref = model._encode_frame(x, lengths)
    captured = []
    hook = model.quantizer.register_forward_pre_hook(lambda m, args: captured.append(args))
    model._encode_frame(x, lengths)
    hook.remove()
    z, n, _ = captured[0]
    counts = [None, 0, 1, 7, 32, 40, -1]
    refs = {count: model.quantizer(z, n, count) for count in counts}
    with pytest.raises(ValueError, match='cached codebooks'):
        with optimized(model, quantizer_backend='triton', cache_codebooks=False): pass
    with optimized(model, residual_backend='triton', quantizer_backend='triton'):
        for count in counts:
            out = model.quantizer(z, n, count)
            for a, b in zip(refs[count], out): assert torch.equal(a, b)
        out = model._encode_frame(x, lengths)
        for key in ref: assert torch.equal(ref[key], out[key])
        assert model.quantizer._fast_codes_only is False
        # A hook must still observe the original quantized vectors; skip the
        # encoder-only dead-output elimination when those vectors are observed.
        observed = []
        hook = model.quantizer.register_forward_hook(lambda m, args, out: observed.append(out))
        model._encode_frame(x, lengths)
        hook.remove()
        for a, b in zip(refs[None], observed[0]): assert torch.equal(a, b)
        assert model.quantizer._fast_codes_only is False
        observed.clear()
        def global_observer(module, args, output):
            if module is model.quantizer: observed.append(output)
        hook = torch.nn.modules.module.register_module_forward_hook(global_observer)
        try:
            model._encode_frame(x, lengths)
        finally:
            hook.remove()
        for a, b in zip(refs[None], observed[0]): assert torch.equal(a, b)
        # Public quantizer graph returns vectors as well as indices and lengths.
        graph = GraphedCallable(lambda h, n: model.quantizer(h, n), z, n)
        for a, b in zip(refs[None], graph(z, n)): assert torch.equal(a, b)
        del graph
    assert model._encode_frame == original_encode
    assert not hasattr(model.quantizer, '_fast_codes_only')
    for key in ref: assert torch.equal(ref[key], model._encode_frame(x, lengths)[key])


@torch.inference_mode()
def test_residual_noncontiguous_fallback_and_flag_cleanup():
    from benchmarks.fixtures import structural_model
    from fast_moss.optimize import optimized
    model = structural_model()
    # Identity input projection deliberately retains gapped input strides.
    model.quantizer.input_proj = torch.nn.Identity()
    z = torch.randn(2, 512, 6, device='cuda')[..., ::2]
    n = torch.tensor([3, 1], device='cuda')
    ref = model.quantizer(z, n)
    with optimized(model, quantizer_backend='triton'):
        for a, b in zip(ref, model.quantizer(z, n)): assert torch.equal(a, b)
        previous = model._fast_original_encode_frame
        def fail(*args, **kwargs): raise RuntimeError('expected failure')
        model._fast_original_encode_frame = fail
        with pytest.raises(RuntimeError, match='expected failure'):
            model._encode_frame(torch.zeros(1, 1, 1920, device='cuda'))
        assert model.quantizer._fast_codes_only is False
        model._fast_original_encode_frame = previous

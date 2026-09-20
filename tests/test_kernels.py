import pytest
import torch
from fast_moss.kernels import scale_add
from fast_moss.cute_kernels import scale_add as cute_scale_add


@pytest.mark.parametrize("kernel", [scale_add, cute_scale_add])
@pytest.mark.parametrize("shape", [(1, 1, 1280), (2, 8, 768), (1, 100, 768), (2, 3, 123)])
def test_residual_bitwise(shape, kernel):
    torch.manual_seed(9)
    x = torch.randn(shape, device="cuda")
    update = torch.randn_like(x)
    scale = torch.randn(shape[-1], device="cuda")
    assert torch.equal(kernel(x, update, scale), x + scale * update)


@pytest.mark.parametrize("kernel", [scale_add, cute_scale_add])
def test_residual_cancellation(kernel):
    # A fused multiply-add would change these results.
    x = torch.tensor([[-1., -1., 1.]], device="cuda")
    u = torch.tensor([[1.0000001192092896, 0.9999999403953552, -1.0000001192092896]], device="cuda")
    s = torch.tensor([0.9999998807907104, 1.0000001192092896, 0.9999998807907104], device="cuda")
    assert torch.equal(kernel(x, u, s), x + s * u)


@pytest.mark.parametrize("kernel", [scale_add, cute_scale_add])
@pytest.mark.parametrize("shape,strides", [
    ((24, 1, 1280), (1280, 1, 1)),
    ((24, 1, 1280), (1280, 7, 1)),
    ((1, 3, 1280), (1, 1280, 1)),
    ((2, 1, 3, 128), (384, 1, 128, 1)),
    ((1, 1, 1280), (1, 1, 1)),
])
def test_residual_singleton_layout_and_downstream_matmul(kernel, shape, strides):
    from fast_moss.graphs import GraphedCallable
    from fast_moss.loading import strict_precision
    strict_precision()
    torch.manual_seed(472)
    x = torch.empty_strided(shape, strides, device='cuda').normal_()
    update = torch.randn(shape, device='cuda')
    scale = torch.randn(shape[-1], device='cuda')
    assert x.is_contiguous()
    reference = x + update * scale
    candidate = kernel(x, update, scale)
    assert candidate.stride() == reference.stride()
    assert torch.equal(candidate, reference)
    # Layout determines whether F.linear folds higher-rank inputs into GEMM.
    weight = torch.randn(768, shape[-1], device='cuda')
    expected = torch.nn.functional.linear(reference, weight)
    assert torch.equal(torch.nn.functional.linear(candidate, weight), expected)
    graph = GraphedCallable(lambda z: (kernel(z, update, scale),), x)
    replay = graph(x)[0]
    assert replay.stride() == reference.stride()
    assert torch.equal(torch.nn.functional.linear(replay, weight), expected)

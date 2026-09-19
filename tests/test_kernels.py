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

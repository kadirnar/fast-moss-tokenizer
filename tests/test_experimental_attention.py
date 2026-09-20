"""Numerical sanity checks for research code, not a model-quality approval gate."""
import pytest
import torch
import torch.nn.functional as F
from benchmarks.experimental_attention import split_attention
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize("length,scale", [(1, 1.), (125, 1.), (251, 20.), (1007, 1.)])
@pytest.mark.parametrize("block", [64, 128, 256])
def test_split_attention_tails_strides_empty_rows(length, scale, block):
    torch.manual_seed(771)
    # Packed projection layout for queries and noncontiguous K/V feature stride.
    q = torch.randn(2, 3, 3, 2, 64, device="cuda").permute(2, 0, 3, 1, 4)[0] * scale
    k = torch.randn(2, 2, length, 128, device="cuda")[..., ::2] * scale
    v = torch.randn(2, 2, length, 128, device="cuda")[..., ::2]
    bias = torch.zeros(2, 1, 3, length + 7, device="cuda")[..., :length]
    bias[0, :, 0] = -float("inf")
    bias[1, :, 1, length//2:] = -float("inf")
    expected = F.scaled_dot_product_attention(q, k, v, bias)
    actual = split_attention(q, k, v, bias, block)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(actual[0, :, 0]) == 0
    graph = GraphedCallable(lambda q, k, v, bias: (split_attention(q, k, v, bias, block),), q, k, v, bias)
    assert torch.equal(graph(q, k, v, bias)[0], actual)

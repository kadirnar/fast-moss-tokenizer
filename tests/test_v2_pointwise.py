"""Mixed precision rounding, layout and ownership gates for v2 pointwise fusion."""

import pytest
import torch

from fast_moss.graphs import GraphedCallable
from fast_moss.v2 import optimized
from fast_moss.v2_pointwise import rotate, scale_add
from test_v2_runtime import model


def raw_equal(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    dtype = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(a.view(dtype), b.view(dtype))


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 12, 1, 64), (2, 12, 24, 64), (1, 20, 129, 64)])
@pytest.mark.parametrize("layout", ["contiguous", "qkv"])
def test_rotary_native_roundings_changed_graph_inputs(dtype, shape, layout):
    b, h, t, d = shape
    torch.manual_seed(64048)
    if layout == "qkv":
        packed = torch.randn(b, t, 3, h, d, device="cuda", dtype=dtype)
        q, k = packed.permute(2, 0, 3, 1, 4)[:2]
    else:
        q = torch.randn(shape, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
    phase = torch.randn(b, t, d // 2, device="cuda") * 10000
    cos, sin = phase.cos(), phase.sin()

    def native(q, k):
        qr, qi = q[..., ::2].float(), q[..., 1::2].float()
        kr, ki = k[..., ::2].float(), k[..., 1::2].float()
        c, s = cos[:, None], sin[:, None]
        return (
            torch.stack(
                [(qr * c - qi * s).to(dtype), (qr * s + qi * c).to(dtype)], -1
            ).reshape(shape),
            torch.stack(
                [(kr * c - ki * s).to(dtype), (kr * s + ki * c).to(dtype)], -1
            ).reshape(shape),
        )

    graph = GraphedCallable(lambda a, b: rotate(a, b, cos, sin), q, k)
    for a, b in [
        (q, k),
        (q * 1.0e-38, k * 1.0e-38),
        (q * 1.0e30, k * 1.0e30),
        (torch.full_like(q, -0.0), torch.zeros_like(k)),
        (torch.full_like(q, float("nan")), torch.full_like(k, float("inf"))),
    ]:
        for result in [rotate(a, b, cos, sin), graph(a, b)]:
            for out, ref in zip(result, native(a, b)):
                raw_equal(out, ref)


@torch.inference_mode()
@pytest.mark.parametrize("residual_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("update_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_scaled_residual_native_roundings(residual_dtype, update_dtype, layout):
    torch.manual_seed(164048)
    x = torch.randn(2, 3, 768, device="cuda", dtype=residual_dtype)
    u = torch.randn(x.shape, device="cuda", dtype=update_dtype)
    scale = torch.randn(768, device="cuda") * 0.01
    if layout == "strided":
        x = x[:, ::2]
        u = u[:, ::2]
    graph = GraphedCallable(lambda a, b: (scale_add(a, b, scale),), x, u)
    for a, b in [
        (x, u),
        (x * 1.0e-38, u * 1.0e-38),
        (x * 1.0e30, u * 1.0e30),
        (torch.full_like(x, -0.0), torch.zeros_like(u)),
        (torch.full_like(x, float("nan")), torch.full_like(u, float("inf"))),
    ]:
        expected = a.float() + scale * b
        for result in [scale_add(a, b, scale), graph(a, b)[0]]:
            raw_equal(result, expected)


@torch.inference_mode()
def test_v2_pointwise_real_module_offsets_and_hooks(model):
    layer = next(
        m
        for m in model.modules()
        if type(m).__name__ == "MossAudioTokenizerTransformerLayer"
    )
    rope = layer.self_attn.rope
    original_rope = rope.forward
    original_layer = layer.forward
    q = torch.randn(2, 1, 3, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    x = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([3, 2], device="cuda")
    with optimized(model) as owner:
        for offset in [
            torch.tensor([0, 0], device="cuda"),
            torch.tensor([12345, 67], device="cuda"),
        ]:
            for out, ref in zip(rope(q, k, offset), original_rope(q, k, offset)):
                raw_equal(out, ref)
        assert owner.rope_calls == 2
        rope.max_period = 1000
        for out, ref in zip(rope(q, k, offset), original_rope(q, k, offset)):
            raw_equal(out, ref)
        assert owner.rope_calls == 2
        rope.max_period = 10000
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = original_layer(x, input_lengths=lengths)
            actual = layer(x, input_lengths=lengths)
            raw_equal(actual, expected)
            assert owner.residual_calls == 2
            calls = []
            hook = layer.layer_scale_1.register_forward_hook(
                lambda *args: calls.append(1)
            )
            try:
                actual = layer(x, input_lengths=lengths)
                assert len(calls) == 1
                raw_equal(actual, expected)
                assert owner.residual_calls == 2
            finally:
                hook.remove()
    assert rope.forward == original_rope and layer.forward == original_layer
    assert not owner.frequencies


def test_v2_pointwise_gradient_fallback(model):
    layer = next(
        m
        for m in model.modules()
        if type(m).__name__ == "MossAudioTokenizerTransformerLayer"
    )
    x = torch.randn(1, 2, 64, device="cuda", requires_grad=True)
    lengths = torch.tensor([2], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = torch.autograd.grad(layer(x, input_lengths=lengths).sum(), x)[0]
        with optimized(model) as owner:
            actual = torch.autograd.grad(layer(x, input_lengths=lengths).sum(), x)[0]
            raw_equal(reference, actual)
            assert owner.residual_calls == 0 and owner.rope_calls == 0

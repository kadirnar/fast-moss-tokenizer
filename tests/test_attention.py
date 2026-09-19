import pytest
import torch
from unittest.mock import patch
from fast_moss.attention import causal_mask
from fast_moss.optimize import optimized
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformer


@pytest.mark.parametrize("context", [None, 1, 7, 1024])
@pytest.mark.parametrize("offset_value", [0, 17, 2**40])
def test_mask_matches_integer_reference(context, offset_value):
    offset = torch.tensor([offset_value, offset_value + 11], device="cuda", dtype=torch.long)
    # Noncontiguous positions include invalid history, wrap order, and future keys.
    pos = (torch.arange(38, device="cuda")[None, :] + offset[:, None] - 20)[:, ::2]
    pos[:, 0] = -1
    delta = offset[:, None, None] + torch.arange(5, device="cuda")[None, :, None] - pos[:, None]
    expected = (pos[:, None] >= 0) & (delta >= 0)
    if context is not None:
        expected &= delta < context
    assert torch.equal(causal_mask(offset, pos, 5, context), expected[:, None])
    additive = causal_mask(offset, pos, 5, context, additive=True)
    assert torch.equal(additive, torch.where(expected[:, None], 0.0, -float("inf")))
    assert all(stride % 8 == 0 for stride in additive.stride()[:-1])
    expanded = pos[:1].expand(2, -1)
    delta = offset[:, None, None] + torch.arange(5, device="cuda")[None, :, None] - expanded[:, None]
    expected = (expanded[:, None] >= 0) & (delta >= 0)
    if context is not None:
        expected &= delta < context
    assert torch.equal(causal_mask(offset, expanded, 5, context), expected[:, None])


@torch.inference_mode()
@pytest.mark.parametrize("causal,context", [(True, None), (True, 7), (False, None)])
def test_shared_mask_exact_and_constructed_once(causal, context):
    stage = MossAudioTokenizerTransformer(64, 1, num_layers=3, dim_feedforward=128,
                                         causal=causal, context=context, positional_embedding="sin")
    stage = stage.cuda().eval().requires_grad_(False)
    x = torch.randn(2, 11, 64, device="cuda")
    reference = stage(x)
    with optimized(stage, attention_mask_backend="triton"):
        with patch("fast_moss.attention.causal_mask", wraps=causal_mask) as construction:
            actual = stage(x)
            assert construction.call_count == int(causal)
        assert torch.equal(actual, reference)
        assert all(layer.self_attn._fast_mask_pool is None for layer in stage.layers)

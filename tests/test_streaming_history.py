"""Ensure insertion of a chunk cannot evict history its first query still needs."""
import torch
from upstream.modeling_moss_audio_tokenizer import RingKVCache
from fast_moss.streaming import StreamingSession
from benchmarks.fixtures import structural_model


def _attend(cache, values, offset, context):
    keys = torch.zeros_like(values)
    _, cached_values, positions = cache.complete(keys, values, torch.ones(1, device="cuda", dtype=torch.bool))
    qpos = torch.arange(values.shape[2], device="cuda")[None, :, None] + offset
    delta = qpos - positions[:, None, :]
    mask = (positions[:, None, :] >= 0) & (delta >= 0) & (delta < context)
    weights = mask.float() / mask.sum(-1, keepdim=True)
    return weights @ cached_values[:, 0]


def test_chunk_insertion_history_requirement():
    # Exact uniform-attention oracle: query at position 3 must average values 1,2,3.
    context, chunk = 3, 2
    cache = RingKVCache(1, 1, 1, context + chunk - 1, dtype=torch.float32)
    _attend(cache, torch.tensor([[[[1.], [2.]]]], device="cuda"), 0, context)
    result = _attend(cache, torch.tensor([[[[3.], [4.]]]], device="cuda"), 2, context)
    torch.testing.assert_close(result, torch.tensor([[[2.], [3.]]], device="cuda"), rtol=0, atol=0)
    # Demonstrate the original capacity loses the oldest still-required key.
    legacy = RingKVCache(1, 1, 1, context, dtype=torch.float32)
    _attend(legacy, torch.tensor([[[[1.], [2.]]]], device="cuda"), 0, context)
    wrong = _attend(legacy, torch.tensor([[[[3.], [4.]]]], device="cuda"), 2, context)
    assert wrong[0, 0, 0].item() == 2.5


@torch.inference_mode()
def test_session_reserves_history_at_every_resolution():
    model = structural_model()
    for direction in ("encode", "decode"):
        with StreamingSession(model, direction, chunk_frames=2, use_graph=False) as session:
            tokens = 3840 if direction == "encode" else 2
            for root in session.modules:
                for child in root.modules():
                    state = getattr(child, "_streaming_state", None)
                    cache = getattr(state, "kv_cache", None)
                    if cache is not None:
                        assert cache.capacity == child.context + tokens - 1
                if hasattr(root, "patch_size"):
                    tokens = tokens // root.patch_size if direction == "encode" else tokens * root.patch_size

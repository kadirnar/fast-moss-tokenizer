"""Stage-local reuse of position-derived tensors, with streaming invariants."""
import torch
from .rope import tables


def stage_forward(self, x, *args, **kwargs):
    state = self._streaming_state
    if (x.dtype != torch.float32 or not x.is_cuda
            or (state is not None and not getattr(state, "_fast_synchronized", False))):
        return self._fast_original_stage(x, *args, **kwargs)
    saved = []
    try:
        if self._fast_share_rope and self.positional_embedding == "rope":
            b, t, _ = x.shape
            attn = self.layers[0].self_attn
            offset = (torch.zeros(b, device=x.device, dtype=torch.long) if state is None
                      else attn._streaming_state.offset)
            saved.append((self.rope, "_fast_tables", self.rope._fast_tables))
            self.rope._fast_tables = tables(
                self.rope, x.device, b, t, attn.embed_dim // attn.num_heads, offset)
        if self._fast_share_masks:
            pool = {}
            for layer in self.layers:
                attn = layer.self_attn
                saved.append((attn, "_fast_mask_pool", attn._fast_mask_pool))
                attn._fast_mask_pool = pool
        return self._fast_original_stage(x, *args, **kwargs)
    finally:
        for obj, name, previous in reversed(saved):
            setattr(obj, name, previous)

"""Reset selected lanes across all known streaming states with one GPU launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _reset(POINTERS, MASK, ACTIVE, CLOSED, B: tl.constexpr, COUNT: tl.constexpr,
           MS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = i % B
    selected = tl.load(MASK + lane * MS)
    address = tl.load(POINTERS + i // B, i < COUNT * B, 0)
    pointer = address.to(tl.pointer_type(tl.int64))
    tl.store(pointer + lane, 0, (i < COUNT * B) & selected)
    tl.store(ACTIVE + lane, True, (i < B) & selected)
    tl.store(CLOSED + lane, False, (i < B) & selected)


class ResetPlan:
    """Own referenced tensors until session exit; unknown state types fall back."""

    def __init__(self, states, active, closed):
        self.states = states
        self.active, self.closed = active, closed
        self.offsets = []
        seen = set()
        for state in states:
            if type(state).__name__ not in {'StreamingState', 'MHAState', 'LayerState', 'TransformerState'}:
                raise ValueError('Unknown streaming state reset semantics')
            if state.exec_mask is not active:
                raise ValueError('State masks must share the session mask')
            candidates = [getattr(state, name, None) for name in ('offset', 'offsets')]
            cache = getattr(state, 'kv_cache', None)
            if cache is not None:
                candidates.append(cache.end_offset)
            for tensor in candidates:
                if tensor is None:
                    continue
                if (tensor.dtype != torch.long or tensor.shape != active.shape
                        or tensor.device != active.device or not tensor.is_contiguous()):
                    raise ValueError('Expected contiguous per-lane int64 offsets')
                if tensor.data_ptr() not in seen:
                    seen.add(tensor.data_ptr())
                    self.offsets.append(tensor)
        self.pointers = torch.tensor([t.data_ptr() for t in self.offsets], device=active.device, dtype=torch.long)

    def reset(self, mask):
        b = self.active.numel()
        _reset[(triton.cdiv(max(len(self.offsets), 1) * b, 256),)](
            self.pointers, mask, self.active, self.closed, b, len(self.offsets), mask.stride(0), 256)
        for state in self.states:
            if hasattr(state, 'offset_cpu'):
                state.offset_cpu = 0

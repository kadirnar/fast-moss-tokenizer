"""Two-launch ring KV update; inactive lanes keep their cache and position."""
import torch
import triton
import triton.language as tl


@triton.jit
def _write(K, V, CACHE, END, ACTIVE, N: tl.constexpr,
           H: tl.constexpr, T: tl.constexpr, D: tl.constexpr, C: tl.constexpr,
           B: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
           KS3: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
           VS3: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    t = i // D % T
    h = i // (D * T) % H
    b = i // (D * T * H)
    active = tl.load(ACTIVE + b, i < N, 0)
    end = tl.load(END + b, i < N, 0)
    slot = (end + t) % C
    dst = ((b * H + h) * C + slot) * D + d
    mask = (i < N) & active
    k = tl.load(K + b * KS0 + h * KS1 + t * KS2 + d * KS3, mask, 0)
    v = tl.load(V + b * VS0 + h * VS1 + t * VS2 + d * VS3, mask, 0)
    tl.store(CACHE + dst, k, mask)
    tl.store(CACHE + B * H * C * D + dst, v, mask)


@triton.jit
def _advance(END, ACTIVE, POS, C: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    end = tl.load(END + b)
    active = tl.load(ACTIVE + b)
    new_end = end + tl.where(active, T, 0)
    last = new_end - 1
    delta = i - last % C
    positions = tl.where(delta <= 0, last + delta, last + delta - C)
    positions = tl.where(i >= new_end, -1, positions)
    tl.store(POS + b * C + i, positions, i < C)
    tl.store(END + b, new_end)


def complete(cache, k, v, active):
    """Update a contiguous upstream RingKVCache without changing active-lane math."""
    b, h, t, d = k.shape
    if t < 1 or t > cache.capacity:
        raise ValueError("Cache update length must be within [1, capacity]")
    if not cache.respect_exec_mask or cache.end_offset.shape != (b,):
        raise ValueError("This kernel requires per-lane offsets")
    if (k.shape != v.shape or k.dtype != v.dtype or k.dtype != cache.cache.dtype
            or cache.cache.shape != (2, b, h, cache.capacity, d)):
        raise ValueError("KV dimensions and dtype must match the cache")
    if (not k.is_cuda or not all(x.device == k.device for x in (v, cache.cache, active, cache.end_offset))
            or active.shape != (b,) or active.dtype != torch.bool or not active.is_contiguous()
            or not cache.cache.is_contiguous() or not cache.end_offset.is_contiguous()):
        raise ValueError("Invalid CUDA cache buffers or execution mask")
    positions = torch.empty(b, cache.capacity, device=k.device, dtype=torch.long)
    _write[(triton.cdiv(k.numel(), 256),)](
        k, v, cache.cache, cache.end_offset, active, k.numel(), h, t, d, cache.capacity,
        b, *k.stride(), *v.stride(), 256)
    _advance[(b,)](cache.end_offset, active, positions, cache.capacity, t,
                   triton.next_power_of_2(cache.capacity))
    return cache.cache[0], cache.cache[1], positions


def attention_complete(self, k, v):
    state = self._streaming_state
    if state is None or state.kv_cache is None:
        return self._fast_original_complete(k, v)
    return complete(state.kv_cache, k, v, state.exec_mask)

"""Research-only block exponent packing; every FP32 bit is reconstructed.

This is an integer storage codec, not a lower-precision floating-point format.
It preserves sign, mantissa, exponent, signed zeros, infinities, and NaN payloads.
"""
from dataclasses import dataclass
import torch
import triton
import triton.language as tl


@triton.jit
def _headers(X, Header, Words, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    i = block * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + i, i < N, 0).to(tl.uint32)
    exponent = (value >> 23) & 255
    lo = tl.min(tl.where(i < N, exponent, 255), 0)
    hi = tl.max(tl.where(i < N, exponent, 0), 0)
    span = hi - lo
    bits = tl.full((), 0, tl.int32)
    for b in tl.static_range(8):
        bits += (span >= (1 << b)).to(tl.int32)
    tl.store(Header + block, lo | (bits << 8))
    tl.store(Words + block, (24 + bits) * (BLOCK // 32))


@triton.jit
def _code(X, index, N: tl.constexpr, BASE):
    value = tl.load(X + index, index < N, 0).to(tl.uint32)
    payload = (value & 0x7fffff) | ((value >> 8) & 0x800000)
    exponent = ((value >> 23) & 255) - BASE
    return payload | (exponent << 24)


@triton.jit
def _pack(X, Header, Offsets, Packed, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    header = tl.load(Header + block)
    base = header & 255
    width = 24 + (header >> 8)
    j = tl.arange(0, BLOCK)  # Up to BLOCK words for width=32.
    bit = j * 32
    index = bit // width
    shift = bit % width
    start = block * BLOCK
    first = _code(X, start + index, N, base)
    second = _code(X, start + index + 1, N, base)
    third = _code(X, start + index + 2, N, base)
    out = first >> shift
    second_shift = width - shift
    out |= tl.where(second_shift < 32, second << second_shift, 0)
    third_shift = 2 * width - shift
    out |= tl.where(third_shift < 32, third << third_shift, 0)
    offset = tl.load(Offsets + block)
    tl.store(Packed + offset + j, out, j < width * (BLOCK // 32))


@triton.jit
def _unpack(Packed, Header, Offsets, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    header = tl.load(Header + block)
    base = header & 255
    width = 24 + (header >> 8)
    offset = tl.load(Offsets + block)
    j = tl.arange(0, BLOCK)
    bit = j * width
    index, shift = bit // 32, bit % 32
    # Request low-priority caching for the compressed input stream.
    first = tl.load(Packed + offset + index, eviction_policy='evict_first').to(tl.uint32)
    second = tl.load(Packed + offset + index + 1,
                     (shift != 0) & (index + 1 < width * (BLOCK // 32)), 0,
                     eviction_policy='evict_first').to(tl.uint32)
    code = (first >> shift) | tl.where(shift != 0, second << (32 - shift), 0)
    # The upper bits can belong to the following value when width < 32.
    delta = (code >> 24) & ((1 << (width - 24)) - 1)
    bits = (code & 0x7fffff) | ((code & 0x800000) << 8) | ((base + delta) << 23)
    i = block * BLOCK + j
    tl.store(Out + i, bits, i < N)


@dataclass
class PackedFloat32:
    data: torch.Tensor
    headers: torch.Tensor
    offsets: torch.Tensor
    shape: tuple
    block: int

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.data, self.headers, self.offsets))

    def unpack(self, out=None):
        n = 1
        for dimension in self.shape: n *= dimension
        if out is None:
            out = torch.empty(self.shape, device=self.data.device, dtype=torch.float32)
        if (out.shape != self.shape or out.dtype != torch.float32 or out.device != self.data.device
                or not out.is_contiguous()):
            raise ValueError('Expected a contiguous same-device FP32 output with the original shape')
        if n:
            _unpack[(triton.cdiv(n, self.block),)](self.data, self.headers, self.offsets, out.view(torch.int32), n, self.block)
        return out


def pack(value, block=256):
    """Offline GPU packing; prefix-sum size read synchronizes and cannot capture."""
    if (not value.is_cuda or value.dtype != torch.float32 or not value.is_contiguous()
            or block not in (128, 256, 512, 1024)):
        raise ValueError('Expected contiguous CUDA FP32 storage and a supported block size')
    n = value.numel()
    if n >= 2**31: raise ValueError('Tensor exceeds 32-bit indexing')
    blocks = triton.cdiv(n, block)
    headers = torch.empty(blocks, device=value.device, dtype=torch.int32)
    words = torch.empty_like(headers)
    if n:
        _headers[(blocks,)](value.view(torch.int32), headers, words, n, block)
        cumulative = words.to(torch.int64).cumsum(0)
        total = int(cumulative[-1])
        if total >= 2**31: raise ValueError('Packed storage exceeds 32-bit word offsets')
        offsets = (cumulative - words).to(torch.int32)
    else:
        total = 0; offsets = torch.empty_like(headers)
    data = torch.empty(total, device=value.device, dtype=torch.int32)
    if n:
        _pack[(blocks,)](value.view(torch.int32), headers, offsets, data, n, block)
    return PackedFloat32(data, headers, offsets, tuple(value.shape), block)

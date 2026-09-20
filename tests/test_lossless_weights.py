import pytest
import torch
from benchmarks.lossless_weights import pack
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('block', [128, 256, 512, 1024])
@pytest.mark.parametrize('length', [0, 1, 257, 65539])
def test_lossless_fp32_bit_patterns_and_graph(block, length):
    torch.manual_seed(137)
    # Uniform raw bits include every exponent class and arbitrary NaN payloads.
    raw = torch.randint(-(2**31), 2**31 - 1, (length,), dtype=torch.int32, device='cuda')
    value = raw.view(torch.float32)
    compressed = pack(value, block)
    actual = compressed.unpack()
    assert torch.equal(raw, actual.view(torch.int32))
    if length:
        graph = GraphedCallable(lambda dummy: (compressed.unpack(),), torch.zeros(1, device='cuda'))
        assert torch.equal(raw, graph(torch.zeros(1, device='cuda'))[0].view(torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize('bits', range(9))
def test_exponent_widths_signed_zero_and_nan(bits):
    n = 256 * 3 + 7
    i = torch.arange(n, device='cuda', dtype=torch.int32)
    span = (1 << bits) - 1
    base = 0 if bits == 8 else 113
    exponents = base + (i % (span + 1))
    raw = ((i % 2) << 31) | (exponents << 23) | ((i * 9161) & 0x7fffff)
    compressed = pack(raw.view(torch.float32))
    assert torch.equal(raw, compressed.unpack().view(torch.int32))
    assert ((compressed.headers[:3] >> 8) == bits).all()
    if bits < 8: assert compressed.nbytes < raw.numel() * 4 + 256 * 4


def test_lossless_rejects_bad_storage():
    with pytest.raises(ValueError): pack(torch.zeros(256))
    with pytest.raises(ValueError): pack(torch.zeros(256, device='cuda', dtype=torch.float16))
    with pytest.raises(ValueError): pack(torch.zeros(256, device='cuda'), block=64)


@torch.inference_mode()
def test_special_fp32_payloads_are_bitwise_preserved():
    # Include signed zero, infinities, quiet/signaling NaN payloads and subnormals.
    raw = torch.tensor([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00000,
                        0x7fa00001, 0xffc12345, 1, 0x80000001, 0x007fffff],
                       device='cuda', dtype=torch.int64).to(torch.int32).repeat(31)
    compressed = pack(raw.view(torch.float32))
    assert torch.equal(raw, compressed.unpack().view(torch.int32))

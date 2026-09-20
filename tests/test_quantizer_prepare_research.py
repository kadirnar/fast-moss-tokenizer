import pytest
import torch

from benchmarks.quantizer_prepare import prepare, reference
from fast_moss.graphs import GraphedCallable


def check(x, result):
    for actual, expected in zip(result, reference(x)):
        assert actual.dtype == expected.dtype
        assert actual.stride() == expected.stride()
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize('batch,time', [(1,1),(1,3),(1,40),(1,129),(1,1025),
                                      (2,1),(8,3),(128,3),(8,40),(257,17)])
@pytest.mark.parametrize('offset', [0,1])
def test_prepare_reduction_order_and_changed_graph_inputs(batch, time, offset):
    torch.manual_seed(1520)
    source = torch.empty(batch*8*time+offset, device='cuda')[offset:].reshape(batch,8,time)
    source.normal_()
    graph = GraphedCallable(prepare, source)
    values = [source*scale for scale in [0.,1.,1.e-38,1.e-20,1.e-12,1.e18,1.e20]]
    values.extend([torch.full_like(source, -0.), source.clone()])
    for value in values:
        source.copy_(value)
        check(source, prepare(source))
        check(source, graph(source))


@torch.inference_mode()
@pytest.mark.parametrize('batch,time', [(1,1),(1,3),(8,3)])
def test_prepare_nonfinite_rows(batch, time):
    x = torch.ones(batch,8,time,device='cuda')
    for index in range(8):
        for value in [float('inf'), -float('inf'), float('nan')]:
            x.fill_(1.)
            x[:,index,:] = value
            check(x, prepare(x))


@torch.inference_mode()
def test_prepare_rejects_unsupported_inputs_and_preserves_explicit_dtype():
    x = torch.ones(1,8,3,device='cuda')
    for invalid in [x.cpu(), x.half(), x[:,::2], x.transpose(1,2),
                    torch.zeros(0,8,3,device='cuda'), x[:,:,:0]]:
        with pytest.raises(ValueError, match='contiguous CUDA FP32'):
            prepare(invalid)
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        check(x, prepare(x))
    finally:
        torch.set_default_dtype(dtype)

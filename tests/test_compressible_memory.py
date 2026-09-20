import gc
import weakref
import pytest
import torch
from benchmarks.compressible_memory import Allocation
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('compressed', [False, True])
def test_verified_vmm_allocation_bits_graph_and_storage_lifetime(compressed):
    torch.manual_seed(257)
    owner = Allocation((257, 129), compressed)
    reference = torch.randint(-(2**31), 2**31-1, (257, 129), device='cuda', dtype=torch.int32)
    value = owner.tensor()
    assert value.data_ptr() == owner.pointer.value
    assert owner.compression_type == int(compressed)
    value.view(torch.int32).copy_(reference)
    assert torch.equal(value.view(torch.int32), reference)
    weak = weakref.ref(owner)
    del owner; gc.collect()
    assert weak() is not None  # Tensor storage must keep the VMM allocation alive.
    graph = GraphedCallable(lambda dummy: (value.view(torch.int32).clone(),), torch.zeros(1, device='cuda'))
    assert torch.equal(graph(torch.zeros(1, device='cuda'))[0], reference)
    del graph, value; gc.collect()
    assert weak() is None

"""Bit-preserving storage and native arithmetic gates for interleaved GEMV."""
import pytest
import torch
import torch.nn.functional as F
from benchmarks.gemv_interleaved import pack,unpack,gemv
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision


@pytest.mark.parametrize('lanes',[8,16,32])
@pytest.mark.parametrize('group',[1,2,4,8,16,32])
@torch.inference_mode()
def test_raw_ieee_words_round_trip(lanes,group):
    torch.manual_seed(94320)
    raw=torch.randint(-(1<<31),1<<31,(32,lanes*8),dtype=torch.int32,device='cuda')
    special=torch.tensor([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
        0x7f800000,0xff800000,0x7fc00001,0xffc00123,0x7f800001],dtype=torch.int64,device='cuda').int()
    raw.flatten()[:special.numel()]=special
    p=pack(raw.view(torch.float32),lanes,group)
    assert p.data.numel()==raw.numel()
    assert torch.equal(unpack(p).view(torch.int32),raw)
    assert torch.equal(raw.flatten()[:special.numel()],special)
    # Non-contiguous source views are copied in logical order too.
    source=raw.T.contiguous().T
    assert not source.is_contiguous()
    assert torch.equal(unpack(pack(source.view(torch.float32),lanes,group)).view(torch.int32),raw)


@pytest.mark.parametrize('shape',[(768,1280,32),(1280,768,16),(3840,1280,8)])
@pytest.mark.parametrize('group',[2,8,32])
@torch.inference_mode()
def test_interleaved_native_reduction_and_graph(shape,group):
    strict_precision();torch.manual_seed(82930)
    n,k,lanes=shape;w=torch.randn(n,k,device='cuda')*.02
    for weight,inputs in [(w,[torch.randn(1,k,device='cuda'),torch.randn(1,k,device='cuda')*1e-38,
            torch.full((1,k),-0.,device='cuda')]),
        (torch.full_like(w,.125),[torch.full((1,k),-1.401298464324817e-45,device='cuda')])]:
        p=pack(weight,lanes,group);graph=GraphedCallable(lambda z:(gemv(z,p),),inputs[0])
        for x in inputs:
            ref=F.linear(x,weight).view(torch.int32)
            assert torch.equal(gemv(x,p).view(torch.int32),ref)
            assert torch.equal(graph(x)[0].view(torch.int32),ref)
        del graph


@torch.inference_mode()
def test_invalid_storage_contract():
    w=torch.randn(32,128,device='cuda');x=torch.randn(1,128,device='cuda')
    for weight,lanes,group in [(w.cpu(),8,4),(w.double(),8,4),(w,4,4),
                               (w,8,3),(w[:31],8,4),(w[:,:127],8,4)]:
        with pytest.raises(ValueError):pack(weight,lanes,group)
    p=pack(w,8,4)
    for inp,warps,unroll in [(x.double(),1,4),(x.cpu(),1,4),(x[:,:64],1,4),
                            (x,3,4),(x,1,8)]:
        with pytest.raises(ValueError):gemv(inp,p,warps,unroll)

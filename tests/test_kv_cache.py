import pytest
import torch
from fast_moss.kv_cache import complete
from fast_moss.graphs import GraphedCallable
from upstream.modeling_moss_audio_tokenizer import RingKVCache


@torch.inference_mode()
@pytest.mark.parametrize("shape,capacity", [((1, 20, 1, 64), 125), ((2, 12, 8, 64), 23),
                                           ((3, 2, 3, 8), 7)])
def test_fused_cache_matches_reference_through_wrap(shape, capacity):
    b,h,t,d = shape
    ref = RingKVCache(b,h,d,capacity,dtype=torch.float32)
    fast = RingKVCache(b,h,d,capacity,dtype=torch.float32)
    active = torch.ones(b,device="cuda",dtype=torch.bool)
    for step in range(12):
        # Real packed-QKV strides, not merely contiguous test inputs.
        packed = torch.randn(b,t,3,h,d,device="cuda").permute(2,0,3,1,4)
        k,v = packed[1],packed[2]
        a = ref.complete(k,v,active)
        c = complete(fast,k,v,active)
        assert all(torch.equal(x,y) for x,y in zip(a,c))
        assert torch.equal(ref.end_offset, fast.end_offset)


@torch.inference_mode()
def test_fused_cache_pause_resume_and_large_offsets():
    cache = RingKVCache(2,2,8,13,dtype=torch.float32)
    k = torch.randn(2,2,3,8,device="cuda")
    v = torch.randn_like(k)
    active = torch.tensor([True,False],device="cuda")
    cache.end_offset[:] = 2**33
    before = cache.cache.clone()
    complete(cache,k,v,active)
    assert torch.equal(cache.cache[:,1],before[:,1])
    assert cache.end_offset.tolist() == [2**33+3,2**33]
    active[:] = True
    _,_,pos = complete(cache,k,v,active)
    assert pos.max().item() == 2**33+5


@torch.inference_mode()
def test_fused_cache_graph_dynamic_offsets_and_mask():
    cache = RingKVCache(2,1,4,7,dtype=torch.float32)
    k=torch.randn(2,1,2,4,device="cuda")
    v=torch.randn_like(k)
    mask=torch.ones(2,device="cuda",dtype=torch.bool)
    graph=GraphedCallable(lambda a,b,m: complete(cache,a,b,m),k,v,mask)
    cache.end_offset.zero_()
    graph(k,v,mask)
    saved=cache.cache[:,1].clone()
    mask[1]=False
    graph(k+1,v+1,mask)
    assert cache.end_offset.tolist()==[4,2]
    assert torch.equal(cache.cache[:,1],saved)

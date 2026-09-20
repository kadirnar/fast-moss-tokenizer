"""Bit-level regression gates for the research-only lossless storage formats."""
import pytest
import torch
from benchmarks.packed_fp32 import pack,unpack


@pytest.mark.parametrize('mode',['variable','fixed28'])
@torch.inference_mode()
def test_mixed_exponent_widths_and_payload_bits(mode):
    generator=torch.Generator().manual_seed(13513)
    blocks=[]
    for width in [8,3,0,7,4,1,6,2,5]:
        base=0 if width in (0,8) else 31
        exponent=(torch.arange(256,dtype=torch.int64)%max(1,1<<width))+base
        fraction=torch.randint(0,1<<23,(256,),generator=generator,dtype=torch.int64)
        sign=torch.randint(0,2,(256,),generator=generator,dtype=torch.int64)<<31
        raw=sign|(exponent<<23)|fraction
        if width==0:raw[:2]=torch.tensor([0,0x80000000])
        blocks.append(raw.to(torch.int32))
    expected=torch.cat(blocks).reshape(9,256).cuda()
    packed=pack(expected.view(torch.float32),mode=mode)
    actual=unpack(packed).view(torch.int32)
    assert torch.equal(actual,expected)
    # Packing a non-contiguous view must preserve its logical element order.
    transposed=expected.T
    assert torch.equal(unpack(pack(transposed.view(torch.float32),mode=mode)).view(torch.int32),transposed)


@pytest.mark.parametrize('mode',['variable','fixed28'])
@torch.inference_mode()
def test_random_raw_words_and_ieee_special_values(mode):
    generator=torch.Generator().manual_seed(8182)
    raw=torch.randint(-(1<<31),1<<31,(8,256),generator=generator,dtype=torch.int32)
    special=[0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,
             0x7f800000,0xff800000,0x7fc00001,0xffc00123,0x7f800001]
    raw.reshape(-1)[:len(special)]=torch.tensor(special,dtype=torch.int64).to(torch.int32)
    raw=raw.cuda()
    assert torch.equal(unpack(pack(raw.view(torch.float32),mode=mode)).view(torch.int32),raw)


@pytest.mark.parametrize('shape',[(768,1280,32),(1280,768,16),(3840,1280,8)])
@pytest.mark.parametrize('backend',['variable','fixed28','cooperative'])
@torch.inference_mode()
def test_fused_mixed_block_formats_and_signed_zero(shape,backend):
    from fast_moss.loading import strict_precision
    from fast_moss.graphs import GraphedCallable
    from benchmarks.packed_fp32 import gemv as direct
    from benchmarks.packed_fp32_shuffle import gemv as cooperative
    strict_precision();torch.manual_seed(3654)
    n,k,lanes=shape
    w=torch.randn(n,k,device='cuda')*.02
    # Alternate columns mix verbatim and compact blocks within each warp.
    w[::2,::256]=0.
    fn=cooperative if backend=='cooperative' else direct
    config=(lanes,32,4) if backend=='cooperative' else (lanes,2,1,4)
    mode='variable' if backend=='variable' else 'fixed28'
    for weight,x in [(w,torch.randn(1,k,device='cuda')),
                     (torch.full_like(w,.125),torch.full((1,k),-1.401298464324817e-45,device='cuda'))]:
        p=pack(weight,mode=mode);ref=torch.nn.functional.linear(x,weight)
        graph=GraphedCallable(lambda z:(fn(z,p,config),),x)
        assert torch.equal(fn(x,p,config).view(torch.int32),ref.view(torch.int32))
        assert torch.equal(graph(x)[0].view(torch.int32),ref.view(torch.int32))
        del graph

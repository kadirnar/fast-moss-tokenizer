import pytest
import torch
from fast_moss.stream_inputs import prepare


@torch.inference_mode()
@pytest.mark.parametrize('encode',[False,True])
@pytest.mark.parametrize('expanded_mask',[False,True])
def test_prepare_strides_padding_and_closed_lanes(encode,expanded_mask):
    batch=4;width=3840 if encode else 5;q=1 if encode else 32
    shape=(batch,1,width*2) if encode else (q,batch,width*2)
    dtype=torch.float32 if encode else torch.long
    x=torch.arange(torch.tensor(shape).prod().item(),device='cuda').reshape(shape).to(dtype)[...,::2]
    n=[width,1,width,0]
    for b,v in enumerate(n):
        lane=x[b] if encode else x[:,b]
        lane[...,v:]=float('nan') if encode else -999
    raw=torch.tensor(n,device='cuda')
    closed=torch.tensor([False,False,True,False],device='cuda')
    requested=(torch.ones(1,device='cuda',dtype=torch.bool).expand(batch) if expanded_mask
               else torch.tensor([True,False,False,True,True,True,False,True],device='cuda')[::2])
    active=torch.empty(batch,device='cuda',dtype=torch.bool)
    out,lengths=prepare(x,raw,requested,closed,active,encode=encode,width=width,rate=1920)
    expected_active=[bool(n[i]) and bool(requested[i]) and not bool(closed[i]) for i in range(batch)]
    assert active.tolist()==expected_active
    expected=torch.zeros_like(out)
    for b,enabled in enumerate(expected_active):
        if enabled:
            if encode:expected[b,...,:n[b]]=x[b,...,:n[b]]
            else:expected[:,b,:n[b]]=x[:,b,:n[b]]
    assert torch.equal(out,expected)
    assert lengths.tolist()==[((n[i]+1919)//1920*1920 if encode else n[i]) if enabled else 0
                              for i,enabled in enumerate(expected_active)]

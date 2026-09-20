from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from fast_moss.strided_normalization import CONFIGS
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from tests.test_normalization import exact


@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(CONFIGS))
def test_profiled_transposes_values_layout_graph_and_restore(monkeypatch,shape):
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda model:{})
    batch,time,n=shape;torch.manual_seed(346)
    layer=torch.nn.LayerNorm(n).cuda().eval().requires_grad_(False)
    layer.weight.copy_(torch.randn_like(layer.weight));layer.bias.copy_(torch.randn_like(layer.bias))
    model=torch.nn.Sequential(layer).eval();x=torch.randn(batch,n,time,device='cuda').transpose(1,2)
    values=[x,-x,x*1e-38,x*1e20,x*.01+1e4,torch.full_like(x,-0.),torch.full_like(x,float('inf')),torch.full_like(x,float('nan'))]
    references=[layer(v) for v in values];original=layer.forward
    with optimized(model,norm_backend='cuda'):
        runtime=model._fast_norm_runtime
        graph=GraphedCallable(lambda z:(layer(z),),x)
        for z,ref in zip(values,references):
            out=layer(z);exact(out,ref);exact(graph(z)[0],ref)
            assert out.stride()==ref.stride() and out.is_contiguous()
        assert runtime.strided_calls==runtime.calls>0
        before=runtime.calls;runtime.strided_enabled=False
        exact(layer(x),references[0]);assert runtime.calls==before
    assert layer.forward==original
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)


def test_transpose_fallbacks_grad_hooks_and_stream_identity(monkeypatch):
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda model:{})
    layer=torch.nn.LayerNorm(768).cuda().eval().requires_grad_(False);model=torch.nn.Sequential(layer).eval()
    original=layer.forward;x=torch.randn(1,768,4,device='cuda').transpose(1,2)
    badshape=torch.randn(3,768,4,device='cuda').transpose(1,2)
    unaligned=torch.randn(x.numel()+1,device='cuda')[1:].view(1,768,4).transpose(1,2)
    gapped=torch.randn(1,768,8,device='cuda')[:,:,::2].transpose(1,2)
    with optimized(model,norm_backend='cuda'):
        runtime=model._fast_norm_runtime
        for z in [badshape,unaligned,gapped]:exact(layer(z),original(z))
        assert runtime.strided_calls==0
        z=x.detach().requires_grad_(True)
        actual=torch.autograd.grad(layer(z).square().sum(),z)[0]
        expected=torch.autograd.grad(original(z).square().sum(),z)[0];exact(actual,expected)
        with torch.autocast('cuda'):exact(layer(x),original(x))
        assert runtime.strided_calls==0
        expected=original(x)+.125
        handle=layer.register_forward_hook(lambda m,args,out:out+.125)
        try:exact(layer(x),expected)
        finally:handle.remove()
        assert runtime.strided_calls==1
        layer(x.contiguous())
        stream=torch.cuda.current_stream().cuda_stream
        runtime.warmed.discard((stream,(4,768),4))
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm CUDA LayerNorm'):layer(x)
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError,match='owner thread'):pool.submit(layer,x).result()
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            graph=GraphedCallable(lambda z:(layer(z),),x);exact(graph(x)[0],original(x))
        torch.cuda.current_stream().wait_stream(side)
    assert layer.forward==original

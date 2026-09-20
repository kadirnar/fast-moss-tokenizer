import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fast_moss.wide_matrices import CONFIGS,compiler,linear
from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized
from fast_moss.loading import strict_precision
from tests.test_normalization import exact
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer


def profile(monkeypatch):
    strict_precision()
    value=json.loads(Path('fast_moss/matrix_profile.json').read_text())
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda model:value)


@torch.inference_mode()
@pytest.mark.parametrize('shape',sorted(CONFIGS))
def test_wide_native_storage_graph_and_restoration(monkeypatch,shape):
    profile(monkeypatch);torch.manual_seed(674);m,n,k=shape
    layer=torch.nn.Linear(k,n,bias=False).cuda().eval().requires_grad_(False)
    model=torch.nn.Sequential(layer).eval();x=torch.randn(1,m,k,device='cuda');ref=layer(x)
    pointer=layer.weight.data_ptr();original=layer.forward
    with optimized(model,matrix_backend='cuda'):
        owner=model._fast_matrix_runtime
        exact(layer(x),ref);assert owner.wide_calls==1 and owner.packed_bytes==0
        assert layer.weight.data_ptr()==pointer
        graph=GraphedCallable(lambda z:(layer(z),),x);exact(graph(x)[0],ref)
        before=owner.wide_calls;owner.wide_enabled=False;exact(layer(x),ref);assert owner.wide_calls==before
    assert layer.forward==original and layer.weight.data_ptr()==pointer
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)


def test_wide_fallbacks_grad_autocast_alignment_custom_hook(monkeypatch):
    profile(monkeypatch)
    layer=torch.nn.Linear(3072,768,bias=False).cuda().eval().requires_grad_(False);model=torch.nn.Sequential(layer).eval()
    x=torch.randn(4,3072,device='cuda');original=layer.forward
    unaligned=torch.randn(x.numel()+1,device='cuda')[1:].reshape_as(x)
    with optimized(model,matrix_backend='cuda'):
        owner=model._fast_matrix_runtime
        for z in [x[:3],unaligned,x.T.contiguous().T]:exact(layer(z),original(z))
        assert owner.wide_calls==0
        z=x.detach().requires_grad_(True)
        exact(torch.autograd.grad(layer(z).sum(),z)[0],torch.autograd.grad(original(z).sum(),z)[0])
        with torch.autocast('cuda'):assert torch.equal(layer(x),original(x))
        assert owner.wide_calls==0
        expected=original(x)+.125
        handle=layer.register_forward_hook(lambda m,args,out:out+.125)
        try:exact(layer(x),expected)
        finally:handle.remove()
        assert owner.wide_calls==1
    layer.forward=lambda z:original(z)+.25;custom=layer.forward;ref=layer(x)
    with optimized(model,matrix_backend='cuda'):
        exact(layer(x),ref);assert model._fast_matrix_runtime.wide_calls==0
    assert layer.forward is custom


@torch.inference_mode()
@pytest.mark.parametrize('width,rows',[(768,4),(1280,3),(1280,1)])
@pytest.mark.parametrize('residual',['triton','cute'])
def test_wide_coexists_with_ffn_and_norm(monkeypatch,width,rows,residual):
    profile(monkeypatch)
    layer=MossAudioTokenizerTransformerLayer(width,width//64,dim_feedforward=4*width,layer_scale=.1)
    model=torch.nn.Sequential(layer).cuda().eval().requires_grad_(False)
    x=torch.randn(1,rows,width,device='cuda');ref=layer._ff_block(x)
    with optimized(model,matrix_backend='cuda',ffn_backend='triton',residual_backend=residual,norm_backend='cuda'):
        exact(layer._ff_block(x),ref)
        owner=model._fast_matrix_runtime
        assert owner.wide_calls==0
        assert owner.ffn_short_calls==(0 if rows==1 else 2)
        assert owner.ffn_gemv_calls==(1 if rows==1 else 0)
        assert owner.norm_ffn_calls==(1 if rows==1 else 0)
        graph=GraphedCallable(lambda z:(layer._ff_block(z),),x);exact(graph(x)[0],ref)
    exact(layer._ff_block(x),ref)


def test_wide_helper_and_dependency_guards(monkeypatch):
    with pytest.raises(ValueError,match='contiguous CUDA FP32'):linear(torch.randn(4,768),torch.randn(768,768),None)
    monkeypatch.setattr('fast_moss.normalization.version',lambda name:'unvalidated')
    with pytest.raises(ValueError,match='pinned cuda extra'):compiler()

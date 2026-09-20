import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fast_moss.strided_ffn import linear
from fast_moss.short_ffn import PAIRS
GEOMETRIES=[(b,m//b,n) for m,n in sorted(PAIRS) for b in range(1,m) if m%b==0]
from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized
from fast_moss.loading import strict_precision
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer


def fixture(monkeypatch,rows,width,encoder=False):
    strict_precision();torch.manual_seed(938)
    layer=MossAudioTokenizerTransformerLayer(width,width//64,dim_feedforward=4*width,layer_scale=.1)
    if encoder:
        model=torch.nn.Module();model.encoder=torch.nn.Sequential(layer)
    else:model=torch.nn.Sequential(layer)
    model=model.cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text())
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda m:profile)
    return model,layer,torch.randn(1,rows,width,device='cuda')


def exact(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('batch,time,width',GEOMETRIES)
@pytest.mark.parametrize('backend',['triton','cute'])
def test_strided_runtime_exact_storage_graph_disable_and_restore(monkeypatch,batch,time,width,backend):
    model,layer,_=fixture(monkeypatch,batch*time,width,True)
    x=torch.randn(batch,width,time,device="cuda").transpose(1,2);ref=layer._ff_block(x);original=layer._ff_block
    pointers=[m.weight.data_ptr() for m in (layer.linear1,layer.linear2)]
    with optimized(model,matrix_backend='cuda',residual_backend=backend,ffn_backend='triton',norm_backend='cuda'):
        owner=model._fast_matrix_runtime;exact(layer._ff_block(x),ref)
        assert owner.ffn_short_calls==1 and owner.ffn_strided_calls==1 and owner.packed_bytes==0
        assert pointers==[m.weight.data_ptr() for m in (layer.linear1,layer.linear2)]
        graph=GraphedCallable(lambda z:(layer._ff_block(z),),x);exact(graph(x)[0],ref);assert graph(x)[0].stride()==ref.stride()==x.stride()
        before=owner.ffn_strided_calls;owner.ffn_strided_enabled=False;exact(layer._ff_block(x),ref);assert owner.ffn_strided_calls==before
    assert layer._ff_block==original;exact(layer._ff_block(x),ref)
    assert pointers==[m.weight.data_ptr() for m in (layer.linear1,layer.linear2)]
    assert not any(k.startswith('_fast_ffn') for k in layer.__dict__)
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)

@torch.inference_mode()
@pytest.mark.parametrize('rows,width',[(4,768),(3,1280)])
@pytest.mark.parametrize('target',['linear1','linear2','norm2','layer_scale_2','global'])
def test_strided_observers_keep_unfused_values(monkeypatch,rows,width,target):
    model,layer,x=fixture(monkeypatch,rows,width);x=x.transpose(1,2).contiguous().transpose(1,2);seen=[]
    def hook(module,args,out):
        if module in (layer.linear1,layer.linear2,layer.norm2,layer.layer_scale_2):seen.append((id(module),out.clone()))
    handle=torch.nn.modules.module.register_module_forward_hook(hook) if target=='global' else getattr(layer,target).register_forward_hook(hook)
    try:
        ref=layer._ff_block(x);expected=seen.copy();seen.clear()
        with optimized(model,matrix_backend='cuda',residual_backend='triton',ffn_backend='triton',norm_backend='cuda'):
            exact(layer._ff_block(x),ref);assert model._fast_matrix_runtime.ffn_short_calls==0
            assert len(seen)==len(expected)
            for (i,a),(j,b) in zip(seen,expected):assert i==j;exact(a,b)
    finally:handle.remove()

@torch.inference_mode()
@pytest.mark.parametrize('rows,width',[(4,768),(3,1280)])
def test_strided_custom_norm_activation_and_exception_restore(monkeypatch,rows,width):
    model,layer,x=fixture(monkeypatch,rows,width);x=x.transpose(1,2).contiguous().transpose(1,2)
    layer.activation=F.relu;ref=layer._ff_block(x)
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,matrix_backend='cuda',residual_backend='triton',ffn_backend='triton',norm_backend='cuda'):
            exact(layer._ff_block(x),ref);assert model._fast_matrix_runtime.ffn_short_calls==0;raise RuntimeError('body failure')
    assert not hasattr(layer,'_fast_original_ffn')
    layer.activation=F.gelu;original=layer.norm2.forward;layer.norm2.forward=lambda z:original(z)+.01;ref=layer._ff_block(x)
    with optimized(model,matrix_backend='cuda',residual_backend='triton',ffn_backend='triton',norm_backend='cuda'):
        exact(layer._ff_block(x),ref);assert model._fast_matrix_runtime.ffn_short_calls==0


@torch.inference_mode()
def test_strided_layout_fallback(monkeypatch):
    model,layer,_=fixture(monkeypatch,4,768)
    x=torch.randn(1,768,8,device='cuda').transpose(1,2)[:,::2]
    ref=layer._ff_block(x)
    with optimized(model,matrix_backend='cuda',residual_backend='triton',ffn_backend='triton',norm_backend='cuda'):
        exact(layer._ff_block(x),ref);assert model._fast_matrix_runtime.ffn_strided_calls==0

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fast_moss.ffn import linear, math_library
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from fast_moss.optimize import optimized
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer


def fixture(monkeypatch,encoder=False):
    strict_precision();torch.manual_seed(957)
    layer=MossAudioTokenizerTransformerLayer(1280,20,dim_feedforward=5120,layer_scale=.1)
    if encoder:
        model=torch.nn.Module();model.encoder=torch.nn.Sequential(layer)
    else:
        model=torch.nn.Sequential(layer)
    model=model.cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text())
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda model:profile)
    x=torch.randn(8,3,1280,device='cuda')
    return model,layer,x


@torch.inference_mode()
@pytest.mark.parametrize('mode', ['gelu','residual'])
def test_fused_matrix_roundings_extreme_inputs_and_graph(mode):
    strict_precision();torch.manual_seed(846)
    k,n=(1280,5120) if mode=='gelu' else (5120,1280)
    weight=torch.randn(n,k,device='cuda')*.05;packed=weight.T.contiguous()
    residual=torch.randn(24,n,device='cuda');scale=torch.randn(n,device='cuda')
    library=math_library();x=torch.randn(24,k,device='cuda')
    def fn(z):return (linear(z,packed,mode,residual,scale,library),)
    graph=GraphedCallable(fn,x)
    for factor in [0.,1.,1e-38,1e-20,1e20]:
        z=x*factor;raw=F.linear(z,weight)
        ref=F.gelu(raw) if mode=='gelu' else residual+raw*scale
        assert torch.equal(fn(z)[0].view(torch.int32),ref.view(torch.int32))
        assert torch.equal(graph(z)[0],ref)


@torch.inference_mode()
@pytest.mark.parametrize('backend',['triton','cute'])
@pytest.mark.parametrize('encoder',[False,True])
def test_ffn_optimizer_fidelity_layout_fallback_graph_and_restoration(monkeypatch,backend,encoder):
    model,layer,x=fixture(monkeypatch,encoder=encoder)
    reference=layer._ff_block(x)
    fallback=x[:4]
    fallback_reference=layer._ff_block(fallback)
    original_method=layer._ff_block
    weights=[layer.linear1.weight.clone(),layer.linear2.weight.clone()]
    with optimized(model,residual_backend=backend,matrix_backend='triton',ffn_backend='triton'):
        runtime=model._fast_matrix_runtime
        out=layer._ff_block(x)
        assert torch.equal(out,reference) and out.stride()==reference.stride()
        assert runtime.ffn_calls==(0 if encoder else 2)
        assert runtime.ffn_staged_calls==(2 if encoder else 0)
        assert torch.equal(layer._ff_block(fallback),fallback_reference)
        assert runtime.ffn_calls==(0 if encoder else 2)
        assert runtime.ffn_staged_calls==(2 if encoder else 0)
        graph=GraphedCallable(lambda z:(layer._ff_block(z),),x)
        assert torch.equal(graph(x)[0],reference)
        for module,weight in zip([layer.linear1,layer.linear2],weights):
            assert torch.equal(module.weight,weight)
    assert layer._ff_block==original_method
    assert not any(name.startswith('_fast_ffn') or name in {'_fast_original_ffn','_fast_observed_ffn'} for name in layer.__dict__)
    assert torch.equal(layer._ff_block(x),reference)
    assert layer.linear1.weight.is_contiguous() and layer.linear2.weight.is_contiguous()
    with pytest.raises(RuntimeError,match='storage changed'):
        graph(x)


@torch.inference_mode()
@pytest.mark.parametrize('hook_target',['linear1','linear2','norm2','layer_scale_2','global'])
def test_ffn_observers_keep_original_module_values(monkeypatch,hook_target):
    model,layer,x=fixture(monkeypatch)
    seen=[]
    def hook(module,args,out):
        if module in [layer.linear1,layer.linear2,layer.norm2,layer.layer_scale_2]:seen.append((id(module),out.clone()))
    handle=(torch.nn.modules.module.register_module_forward_hook(hook) if hook_target=='global'
            else getattr(layer,hook_target).register_forward_hook(hook))
    try:
        reference=layer._ff_block(x);expected=seen.copy();seen.clear()
        with optimized(model,residual_backend='triton',matrix_backend='triton',ffn_backend='triton'):
            assert torch.equal(layer._ff_block(x),reference)
            assert model._fast_matrix_runtime.ffn_calls==0
            assert len(seen)==len(expected)
            assert all(i==j and torch.equal(a,b) for (i,a),(j,b) in zip(expected,seen))
    finally:
        handle.remove()


@torch.inference_mode()
@pytest.mark.parametrize('one_row',[False,True])
def test_ffn_custom_activation_norm_and_body_exception_cleanup(monkeypatch,one_row):
    model,layer,x=fixture(monkeypatch)
    if one_row:x=x[:1,:1].contiguous()
    layer.activation=lambda value:F.relu(value)
    reference=layer._ff_block(x)
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,residual_backend='triton',matrix_backend='triton',ffn_backend='triton'):
            assert torch.equal(layer._ff_block(x),reference)
            assert model._fast_matrix_runtime.ffn_calls==0
            raise RuntimeError('body failure')
    assert not hasattr(layer,'_fast_original_ffn') and layer.linear1.weight.is_contiguous()
    layer.activation=F.gelu
    norm_forward=layer.norm2.forward
    layer.norm2.forward=lambda value:norm_forward(value)+.01
    reference=layer._ff_block(x)
    with optimized(model,residual_backend='triton',matrix_backend='triton',ffn_backend='triton'):
        assert torch.equal(layer._ff_block(x),reference)
        assert model._fast_matrix_runtime.ffn_calls==0


def test_ffn_library_options_and_helper_validation(monkeypatch):
    model=torch.nn.Linear(1,1).eval().requires_grad_(False)
    for opts in [{'ffn_backend':'bad'},{'ffn_backend':'triton'},
                 {'ffn_backend':'triton','matrix_backend':'triton'}]:
        with pytest.raises(ValueError,match='FFN'):
            with optimized(model,**opts):pass
        assert not hasattr(model,'_fast_optimization_active')
    with pytest.raises(ValueError,match='supported contiguous'):
        linear(torch.randn(24,1280),torch.randn(1280,5120),'gelu',None,None,'bad')
    monkeypatch.setattr('fast_moss.ffn.version',lambda name:'unvalidated')
    with pytest.raises(ValueError,match='validated CUDA'):
        math_library()


@torch.inference_mode()
def test_gelu_special_values_and_math_library_contents(monkeypatch):
    from fast_moss.ffn import _ffn_reduce
    values=torch.tensor([float('inf'),-float('inf'),float('nan'),0.,-0.,1e-45,-1e-45,20.,-20.],device='cuda')
    out=torch.empty_like(values);reference=F.gelu(values)
    _ffn_reduce[(1,)](values,values,values,out,values.numel(),values.numel(),1,'gelu',256,
                      enable_fp_fusion=False,extern_libs={'libdevice':math_library()})
    assert torch.equal(out.isnan(),reference.isnan())
    valid=~reference.isnan()
    assert torch.equal(out[valid].view(torch.int32),reference[valid].view(torch.int32))
    monkeypatch.setattr('fast_moss.ffn.LIBDEVICE_SHA256','invalid')
    with pytest.raises(ValueError,match='validated CUDA'):
        math_library()


@torch.inference_mode()
@pytest.mark.parametrize('mode',['gelu','residual'])
def test_fused_projection_signed_zero_before_epilogue(mode):
    strict_precision()
    k,n=(1280,5120) if mode=='gelu' else (5120,1280)
    x=torch.full((24,k),-1.401298464324817e-45,device='cuda')
    w=torch.full((n,k),.125,device='cuda');packed=w.T.contiguous()
    residual=torch.full((24,n),-0.,device='cuda');scale=torch.ones(n,device='cuda')
    raw=F.linear(x,w);ref=F.gelu(raw) if mode=='gelu' else residual+raw*scale
    library=math_library();fn=lambda z:(linear(z,packed,mode,residual,scale,library),)
    graph=GraphedCallable(fn,x)
    assert torch.equal(ref.view(torch.int32),fn(x)[0].view(torch.int32))
    assert torch.equal(ref.view(torch.int32),graph(x)[0].view(torch.int32))


@torch.inference_mode()
@pytest.mark.parametrize('backend',['triton','cute'])
@pytest.mark.parametrize('encoder',[False,True])
def test_one_row_ffn_fusion_storage_graph_fallback_and_restoration(monkeypatch,backend,encoder):
    model,layer,_=fixture(monkeypatch,encoder=encoder)
    x=torch.randn(1,1,1280,device='cuda');ref=layer._ff_block(x)
    refs={factor:layer._ff_block(x*factor) for factor in (0.,.17,1e-38)}
    original=layer._ff_block
    pointers=[m.weight.data_ptr() for m in (layer.linear1,layer.linear2)]
    with optimized(model,residual_backend=backend,matrix_backend='triton',ffn_backend='triton'):
        runtime=model._fast_matrix_runtime
        out=layer._ff_block(x)
        assert torch.equal(out.view(torch.int32),ref.view(torch.int32)) and out.stride()==ref.stride()
        assert runtime.ffn_gemv_calls==2 and runtime.packed_bytes==0
        assert pointers==[m.weight.data_ptr() for m in (layer.linear1,layer.linear2)]
        graph=GraphedCallable(lambda z:(layer._ff_block(z),),x)
        for factor,expected in refs.items():
            assert torch.equal(graph(x*factor)[0].view(torch.int32),expected.view(torch.int32))
        before=runtime.ffn_gemv_calls
        runtime.ffn_gemv_enabled=False
        assert torch.equal(layer._ff_block(x),ref) and runtime.ffn_gemv_calls==before
    assert layer._ff_block==original and torch.equal(layer._ff_block(x),ref)
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)


@torch.inference_mode()
@pytest.mark.parametrize('hook_target',['linear1','linear2','norm2','layer_scale_2','global'])
def test_one_row_ffn_observers_keep_original_calls(monkeypatch,hook_target):
    model,layer,_=fixture(monkeypatch);x=torch.randn(1,1,1280,device='cuda');seen=[]
    def hook(module,args,out):
        if module in (layer.linear1,layer.linear2,layer.norm2,layer.layer_scale_2):seen.append((id(module),out.clone()))
    handle=(torch.nn.modules.module.register_module_forward_hook(hook) if hook_target=='global'
            else getattr(layer,hook_target).register_forward_hook(hook))
    try:
        ref=layer._ff_block(x);expected=seen.copy();seen.clear()
        with optimized(model,residual_backend='triton',matrix_backend='triton',ffn_backend='triton'):
            assert torch.equal(layer._ff_block(x),ref)
            assert model._fast_matrix_runtime.ffn_gemv_calls==0
            assert len(seen)==len(expected)
            assert all(i==j and torch.equal(a,b) for (i,a),(j,b) in zip(expected,seen))
    finally:handle.remove()

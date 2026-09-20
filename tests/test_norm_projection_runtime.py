import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer

OPTIONS=dict(matrix_backend='cuda',norm_backend='cuda',ffn_backend='triton',residual_backend='triton',attention_mask_backend='triton')

def fixture(monkeypatch):
    strict_precision();torch.manual_seed(970)
    layer=MossAudioTokenizerTransformerLayer(1280,20,dim_feedforward=5120,layer_scale=.1)
    model=torch.nn.Sequential(layer).cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text());monkeypatch.setattr('fast_moss.matrices.load_profile',lambda m:profile)
    return model,layer,torch.randn(1,1,1280,device='cuda')

def exact(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('backend',['triton','cute'])
@pytest.mark.parametrize('block',['_sa_block','_ff_block'])
def test_norm_projection_graph_storage_disable_restore(monkeypatch,backend,block):
    model,layer,x=fixture(monkeypatch);original=getattr(layer,block);ref=original(x)
    pointers=[p.data_ptr() for p in model.parameters()]
    with optimized(model,**dict(OPTIONS,residual_backend=backend)):
        owner=model._fast_matrix_runtime;fn=getattr(layer,block);exact(fn(x),ref)
        assert owner.norm_gemv_calls==1 and owner.packed_bytes==0
        graph=GraphedCallable(lambda z:(fn(z),),x);exact(graph(x)[0],ref)
        for z in [x*1e-20,x*1e10,torch.zeros_like(x),torch.full_like(x,-0.)]:exact(fn(z),original(z));exact(graph(z)[0],original(z))
        before=owner.norm_gemv_calls;owner.norm_gemv_enabled=False;exact(fn(x),ref);assert owner.norm_gemv_calls==before
        assert pointers==[p.data_ptr() for p in model.parameters()]
    assert getattr(layer,block)==original;exact(original(x),ref)
    assert pointers==[p.data_ptr() for p in model.parameters()]
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)
    assert not owner.forwards and not owner.norm_gemv_warmed

@torch.inference_mode()
@pytest.mark.parametrize('block,target',[('_sa_block','norm1'),('_sa_block','qkv'),('_sa_block','self_attn'),('_sa_block','layer_scale_1'),('_sa_block','global'),('_ff_block','norm2'),('_ff_block','linear1'),('_ff_block','linear2'),('_ff_block','layer_scale_2'),('_ff_block','global')])
def test_norm_projection_observer_values(monkeypatch,block,target):
    model,layer,x=fixture(monkeypatch);watched=[layer.norm1,layer.norm2,layer.self_attn,layer.self_attn.in_projs[0],layer.linear1,layer.linear2,layer.layer_scale_1,layer.layer_scale_2];seen=[]
    def hook(module,args,out):
        if module in watched:seen.append((id(module),out.clone()))
    module=layer.self_attn.in_projs[0] if target=='qkv' else getattr(layer,target,None)
    handle=torch.nn.modules.module.register_module_forward_hook(hook) if target=='global' else module.register_forward_hook(hook)
    try:
        ref=getattr(layer,block)(x);expected=seen.copy();seen.clear()
        with optimized(model,**OPTIONS):
            exact(getattr(layer,block)(x),ref);assert model._fast_matrix_runtime.norm_gemv_calls==0
            assert len(seen)==len(expected)
            for (i,a),(j,b) in zip(seen,expected):assert i==j;exact(a,b)
    finally:handle.remove()

@torch.inference_mode()
@pytest.mark.parametrize('block,target',[('_sa_block','norm1'),('_sa_block','qkv'),('_sa_block','self_attn'),('_ff_block','norm2'),('_ff_block','linear1'),('_ff_block','linear2')])
def test_norm_projection_custom_forward_during_context(monkeypatch,block,target):
    model,layer,x=fixture(monkeypatch);original=getattr(layer,block)
    with optimized(model,**OPTIONS):
        module=layer.self_attn.in_projs[0] if target=='qkv' else getattr(layer,target)
        forward=module.forward;module.forward=lambda *args,**kw:forward(*args,**kw)+.125
        ref=original(x);owner=model._fast_matrix_runtime;before=owner.norm_gemv_calls
        exact(getattr(layer,block)(x),ref);assert owner.norm_gemv_calls==before
        module.forward=forward

@pytest.mark.parametrize('block',['_sa_block','_ff_block'])
def test_norm_projection_gradient_autocast_and_layout_fallback(monkeypatch,block):
    model,layer,x=fixture(monkeypatch);original=getattr(layer,block)
    with optimized(model,**OPTIONS):
        fn=getattr(layer,block);owner=model._fast_matrix_runtime
        z=x.detach().requires_grad_(True);a=fn(z);b=original(z)
        exact(a,b);exact(torch.autograd.grad(a.sum(),z)[0],torch.autograd.grad(b.sum(),z)[0])
        with torch.autocast('cuda'):assert torch.equal(fn(x),original(x))
        assert owner.norm_gemv_calls==0
        with torch.inference_mode():
            for z in [torch.randn(1,1280,2,device='cuda').transpose(1,2),
                      torch.randn(1,1,2560,device='cuda')[...,::2]]:
                exact(fn(z),original(z));assert owner.norm_gemv_calls==0

@torch.inference_mode()
def test_norm_projection_exception_cleanup(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._sa_block
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,**OPTIONS):layer(x);raise RuntimeError('body failure')
    assert layer._sa_block==original and not hasattr(layer,'_fast_original_sa')


@torch.inference_mode()
@pytest.mark.parametrize('mode',['none','gelu'])
def test_runtime_kernel_intermediates_and_graph(mode):
    from fast_moss.norm_projection import linear,CONFIGS,SOURCE
    from benchmarks.norm_gemv import SOURCE as research_source
    assert SOURCE==research_source
    strict_precision();torch.manual_seed(971)
    n=3840 if mode=='none' else 5120
    x=torch.randn(1,1280,device='cuda');w=torch.randn(n,1280,device='cuda')*.02
    g=torch.randn(1280,device='cuda');b=torch.randn_like(g)
    def fn(z):return linear(z,w,g,b,1e-5,mode,debug=True)
    graph=GraphedCallable(fn,x)
    for z in [x,x*1e-38,x*1e10,x*.001+10000,torch.full_like(x,-0.)]:
        norm=F.layer_norm(z,(1280,),g,b,1e-5);ref=F.linear(norm,w)
        if mode=='gelu':ref=F.gelu(ref)
        for out,intermediate in (fn(z),graph(z)):exact(out,ref);exact(intermediate,norm)


@torch.inference_mode()
def test_projection_capture_thread_and_inactive_owner(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fast_moss.norm_projection import project
    model,layer,x=fixture(monkeypatch)
    with optimized(model,**OPTIONS):
        owner=model._fast_matrix_runtime
        def run():return project(owner,x,layer.norm1,layer.self_attn.in_projs[0],'none')
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm normalization/projection'):run()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError,match='owner thread'):pool.submit(run).result()
        assert run() is not None
        layer.norm1.eps=1e-4
        assert run() is None
    with pytest.raises(RuntimeError,match='owner thread'):run()


@torch.inference_mode()
def test_norm_projection_packed_transition_fallback(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._sa_block;ref=original(x)
    projection=layer.self_attn.in_projs[0];weight=projection.weight.clone()
    with optimized(model,**OPTIONS):
        owner=model._fast_matrix_runtime
        graph=GraphedCallable(lambda z:(layer._sa_block(z),),x);exact(graph(x)[0],ref)
        projection(torch.randn(1,24,1280,device='cuda'))
        assert not projection.weight.is_contiguous()
        with pytest.raises(RuntimeError,match='storage changed'):graph(x)
        before=owner.norm_gemv_calls
        exact(layer._sa_block(x),ref);assert owner.norm_gemv_calls==before
    assert projection.weight.is_contiguous();exact(projection.weight,weight)
    exact(original(x),ref)

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from fast_moss.attention_residual import CONFIGS
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer

OPTIONS=dict(matrix_backend='cuda',norm_backend='cuda',ffn_backend='triton',residual_backend='triton',attention_mask_backend='triton')

def fixture(monkeypatch,width=1280,rows=1,transpose=False):
    strict_precision();torch.manual_seed(981)
    layer=MossAudioTokenizerTransformerLayer(width,width//64,dim_feedforward=4*width,layer_scale=.1)
    model=torch.nn.Sequential(layer).cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text());monkeypatch.setattr('fast_moss.matrices.load_profile',lambda m:profile)
    x=torch.randn(1,width,rows,device='cuda').transpose(1,2) if transpose else torch.randn(1,rows,width,device='cuda')
    return model,layer,x

def exact(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

GEOMETRIES=[(m,n,t) for m,n,k in sorted(CONFIGS) for t in ([False,True] if m>1 else [True])]

@torch.inference_mode()
@pytest.mark.parametrize('rows,width,transpose',GEOMETRIES)
@pytest.mark.parametrize('backend',['triton','cute'])
def test_attention_runtime_graph_layout_disable_restore(monkeypatch,rows,width,transpose,backend):
    model,layer,x=fixture(monkeypatch,width,rows,transpose);original=layer._sa_block;ref=original(x)
    pointers=[p.data_ptr() for p in model.parameters()]
    with optimized(model,**dict(OPTIONS,residual_backend=backend)):
        owner=model._fast_matrix_runtime;fn=layer._sa_block
        y=fn(x);exact(y,ref);assert y.stride()==ref.stride()
        assert owner.attention_residual_calls==1
        graph=GraphedCallable(lambda z:(fn(z),),x);exact(graph(x)[0],ref)
        for z in [x*1e-20,x*1e10,torch.zeros_like(x),torch.full_like(x,-0.)]:exact(fn(z),original(z));exact(graph(z)[0],original(z))
        before=owner.attention_residual_calls;owner.attention_residual_enabled=False;exact(fn(x),ref);assert owner.attention_residual_calls==before
        assert pointers==[p.data_ptr() for p in model.parameters()]
    exact(original(x),ref);assert layer._sa_block==original
    assert not owner.attention_residual_warmed and not owner.forwards
    assert pointers==[p.data_ptr() for p in model.parameters()]
    with pytest.raises(RuntimeError,match='storage changed'):graph(x)

@torch.inference_mode()
@pytest.mark.parametrize('target',['norm1','input','output','self_attn','layer_scale_1','global'])
@pytest.mark.parametrize('pre',[False,True])
def test_attention_runtime_observer_values(monkeypatch,target,pre):
    model,layer,x=fixture(monkeypatch);seen=[]
    watched=[layer.norm1,layer.self_attn,layer.self_attn.in_projs[0],layer.self_attn.out_projs[0],layer.layer_scale_1]
    def hook(module,args,out=None):
        if module in watched:seen.append((id(module),(args[0] if pre else out).clone()))
    module=layer.self_attn.in_projs[0] if target=='input' else layer.self_attn.out_projs[0] if target=='output' else getattr(layer,target,None)
    if target=='global':
        register=torch.nn.modules.module.register_module_forward_pre_hook if pre else torch.nn.modules.module.register_module_forward_hook
        handle=register(hook)
    else:handle=(module.register_forward_pre_hook if pre else module.register_forward_hook)(hook)
    try:
        ref=layer._sa_block(x);expected=seen.copy();seen.clear()
        with optimized(model,**OPTIONS):
            exact(layer._sa_block(x),ref);assert model._fast_matrix_runtime.attention_residual_calls==0
            assert len(seen)==len(expected)
            for (i,a),(j,b) in zip(seen,expected):assert i==j;exact(a,b)
    finally:handle.remove()

@torch.inference_mode()
@pytest.mark.parametrize('target',['norm1','input','output','self_attn','layer_scale_1','complete'])
def test_attention_runtime_changed_methods(monkeypatch,target):
    model,layer,x=fixture(monkeypatch);original=layer._sa_block
    with optimized(model,**OPTIONS):
        module=layer.self_attn.in_projs[0] if target=='input' else layer.self_attn.out_projs[0] if target=='output' else layer.self_attn if target=='complete' else getattr(layer,target)
        name='_complete_kv' if target=='complete' else 'forward';saved=getattr(module,name)
        replacement=(lambda *args,**kw:saved(*args,**kw)) if target=='complete' else (lambda *args,**kw:saved(*args,**kw)+.125)
        setattr(module,name,replacement)
        try:
            ref=original(x);before=model._fast_matrix_runtime.attention_residual_calls
            exact(layer._sa_block(x),ref);assert model._fast_matrix_runtime.attention_residual_calls==before
        finally:setattr(module,name,saved)


def test_attention_runtime_gradient_autocast_fallback(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._sa_block
    with optimized(model,**OPTIONS):
        z=x.detach().requires_grad_(True);a=layer._sa_block(z);b=original(z)
        exact(a,b);exact(torch.autograd.grad(a.sum(),z)[0],torch.autograd.grad(b.sum(),z)[0])
        with torch.autocast('cuda'):assert torch.equal(layer._sa_block(x),original(x))
        assert model._fast_matrix_runtime.attention_residual_calls==0

@torch.inference_mode()
def test_attention_runtime_packing_and_exception_cleanup(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._sa_block;ref=original(x)
    projection=layer.self_attn.out_projs[0];weight=projection.weight.clone()
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,**OPTIONS):
            owner=model._fast_matrix_runtime
            graph=GraphedCallable(lambda z:(layer._sa_block(z),),x);exact(graph(x)[0],ref)
            projection(torch.randn(1,128,1280,device='cuda'));assert not projection.weight.is_contiguous()
            with pytest.raises(RuntimeError,match='storage changed'):graph(x)
            before=owner.attention_residual_calls;exact(layer._sa_block(x),ref);assert owner.attention_residual_calls==before
            raise RuntimeError('body failure')
    assert layer._sa_block==original and projection.weight.is_contiguous()
    exact(projection.weight,weight)
    assert not hasattr(layer,'_fast_attention_runtime');exact(original(x),ref)

@torch.inference_mode()
def test_attention_runtime_capture_thread_and_fold_guard(monkeypatch):
    model,layer,x=fixture(monkeypatch);projection=layer.self_attn.out_projs[0]
    with optimized(model,**OPTIONS):
        owner=model._fast_matrix_runtime;projection(x)
        def run(z=x):return projection(z,_fast_epilogue=('attention_residual',x,layer.layer_scale_1.scale,layer._fast_scale_add))
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm attention residual'):run()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError,match='owner thread'):pool.submit(layer._sa_block,x).result()
        run();assert owner.attention_residual_calls==1
        # Contiguous singleton dimensions can still fail should_fold's stride guard.
        z=torch.as_strided(x,x.shape,(1280,1,1));before=owner.attention_residual_calls
        expected=x+torch.nn.functional.linear(z,projection.weight)*layer.layer_scale_1.scale
        exact(run(z),expected);assert owner.attention_residual_calls==before


@torch.inference_mode()
@pytest.mark.parametrize('target',['norm1','input','output','self_attn','layer_scale_1','complete','rope','block'])
def test_attention_runtime_preexisting_custom_methods(monkeypatch,target):
    from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerRotaryEmbedding
    model,layer,x=fixture(monkeypatch,rows=3)
    layer.self_attn.rope=MossAudioTokenizerRotaryEmbedding()
    module=(layer.self_attn.in_projs[0] if target=='input' else layer.self_attn.out_projs[0] if target=='output'
            else layer.self_attn if target=='complete' else layer.self_attn.rope if target=='rope' else layer if target=='block' else getattr(layer,target))
    name='_complete_kv' if target=='complete' else '_sa_block' if target=='block' else 'forward'
    original=getattr(module,name)
    def replacement(*args,**kw):
        out=original(*args,**kw)
        if target=='complete':
            k,v,positions=out
            return k,v+.125,positions
        if target=='rope':return out[0]+.125,out[1]
        return out+.125
    setattr(module,name,replacement)
    ref=layer._sa_block(x)
    with optimized(model,**dict(OPTIONS,rope_backend='triton',kv_backend='triton')):
        exact(layer._sa_block(x),ref)
        assert model._fast_matrix_runtime.attention_residual_calls==0
    assert getattr(module,name) is replacement
    exact(layer._sa_block(x),ref)

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
import torch.nn.functional as F
from fast_moss.residual_async import CONFIG,SOURCE,linear,compile_kernel
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer

OPTIONS=dict(matrix_backend='cuda',norm_backend='cuda',ffn_backend='triton',residual_backend='triton',attention_mask_backend='triton')

def fixture(monkeypatch):
    strict_precision();torch.manual_seed(1011)
    layer=MossAudioTokenizerTransformerLayer(1280,20,dim_feedforward=5120,layer_scale=.1)
    model=torch.nn.Sequential(layer).cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text())
    monkeypatch.setattr('fast_moss.matrices.load_profile',lambda m:profile)
    return model,layer,torch.randn(1,1,1280,device='cuda')

def bits(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
def test_residual_async_selected_binary_and_changed_graph_operands():
    from benchmarks.residual_gemv_async import SOURCE as research_source
    assert SOURCE==research_source
    strict_precision();torch.manual_seed(1012)
    x=torch.randn(1,5120,device='cuda');w=torch.randn(1280,5120,device='cuda')*.03
    r=torch.randn(1,1280,device='cuda');s=torch.randn(1280,device='cuda')*.1
    fn=lambda a,ww,rr,ss:(linear(a,ww,rr,ss),)
    graph=GraphedCallable(fn,x,w,r,s)
    for a,ww,rr,ss in [(x,w,r,s),(-x,w,r,s),(x*1e-38,w,torch.zeros_like(r),torch.ones_like(s)),
        (x.abs()*1e-38,w.abs(),torch.zeros_like(r),torch.ones_like(s)),(x*1e10,w,r,s),
        (torch.full_like(x,-0.),w,torch.full_like(r,-0.),torch.full_like(s,-0.)),
        (x,w,-F.linear(x,w)*s,s),(torch.ones_like(x),torch.full_like(w,1.401298464324817e-45),torch.zeros_like(r),torch.ones_like(s)),(x,w,r,s)]:
        ref=rr+F.linear(a,ww)*ss
        for out in (fn(a,ww,rr,ss)[0],graph(a,ww,rr,ss)[0]):bits(out,ref);assert out.stride()==ref.stride()
    _,_,resources,_=compile_kernel(5120,CONFIG)
    report=json.loads(Path('results/residual_gemv_async_confirm.json').read_text())
    assert resources==next(row['resources'][json.dumps(CONFIG)] for row in report['records'] if row['shape'][2]==5120)

@torch.inference_mode()
@pytest.mark.parametrize('backend',['triton','cute'])
@pytest.mark.parametrize('norm_backend',['none','cuda'])
def test_residual_async_capture_switch_storage_and_restore(monkeypatch,backend,norm_backend):
    model,layer,x=fixture(monkeypatch);original=layer._ff_block;ref=original(x)
    pointers=[p.data_ptr() for p in model.parameters()]
    with optimized(model,**dict(OPTIONS,residual_backend=backend,norm_backend=norm_backend)):
        owner=model._fast_matrix_runtime;fn=layer._ff_block;bits(fn(x),ref)
        assert owner.residual_async_calls==1 and owner.norm_async_calls==int(norm_backend=='cuda')
        staged=GraphedCallable(lambda z:(fn(z),),x);before=owner.residual_async_calls
        owner.residual_async_enabled=False;bits(fn(x),ref);assert owner.residual_async_calls==before
        direct=GraphedCallable(lambda z:(fn(z),),x)
        for z in (x,-x,x*.001+10000):bits(staged(z)[0],original(z));bits(direct(z)[0],original(z))
        owner.residual_async_enabled=True;bits(fn(x),ref);assert owner.residual_async_calls==before+1
        assert [p.data_ptr() for p in model.parameters()]==pointers and owner.packed_bytes==0
    assert not owner.ffn_gemv_warmed and not owner.forwards and layer._ff_block==original
    assert [p.data_ptr() for p in model.parameters()]==pointers
    for graph in (staged,direct):
        with pytest.raises(RuntimeError,match='storage changed'):graph(x)

@torch.inference_mode()
@pytest.mark.parametrize('first_async',[False,True])
def test_residual_async_warmup_is_per_backend_and_stream(monkeypatch,first_async):
    from fast_moss.ffn import math_library
    model,layer,x=fixture(monkeypatch);hidden=torch.randn(1,1,5120,device='cuda');library=math_library()
    with optimized(model,**OPTIONS):
        owner=model._fast_matrix_runtime
        def run():return layer.linear2(hidden,_fast_epilogue=('residual',x,layer.layer_scale_2.scale,library))
        owner.residual_async_enabled=first_async;run()
        owner.residual_async_enabled=not first_async
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm FFN GEMV'):run()
        run()
        parent=torch.cuda.current_stream();side=torch.cuda.Stream();side.wait_stream(parent)
        with torch.cuda.stream(side):
            layer.linear2(hidden)  # Ordinary matrix warmup does not warm either epilogue.
            for asynchronous in (False,True):
                owner.residual_async_enabled=asynchronous
                with monkeypatch.context() as patch:
                    patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
                    with pytest.raises(RuntimeError,match='Warm FFN GEMV'):run()
                run()
        parent.wait_stream(side);torch.cuda.synchronize()
        assert len(owner.ffn_gemv_warmed)==4 and owner.residual_async_calls==2
    assert not owner.ffn_gemv_warmed

@torch.inference_mode()
def test_residual_async_packing_exception_and_thread_guards(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._ff_block;ref=original(x)
    projection=layer.linear2;weight=projection.weight.clone();forward=projection.forward
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,**OPTIONS):
            owner=model._fast_matrix_runtime
            graph=GraphedCallable(lambda z:(layer._ff_block(z),),x)
            with ThreadPoolExecutor(1) as pool:
                with pytest.raises(RuntimeError,match='construction host thread'):pool.submit(projection,torch.randn(1,5120,device='cuda')).result()
            before=owner.residual_async_calls
            projection(torch.randn(1,24,5120,device='cuda'));assert not projection.weight.is_contiguous()
            with pytest.raises(RuntimeError,match='storage changed'):graph(x)
            bits(layer._ff_block(x),ref);assert owner.residual_async_calls==before
            raise RuntimeError('body failure')
    bits(projection.weight,weight);assert projection.weight.is_contiguous() and projection.forward==forward
    assert not owner.ffn_gemv_warmed and not hasattr(model,'_fast_matrix_runtime')

@torch.inference_mode()
@pytest.mark.parametrize('target',['linear1','linear2','norm2','layer_scale_2','global'])
@pytest.mark.parametrize('pre',[False,True])
def test_residual_async_preserves_observer_values(monkeypatch,target,pre):
    model,layer,x=fixture(monkeypatch);watched=[layer.linear1,layer.linear2,layer.norm2,layer.layer_scale_2];seen=[]
    def hook(module,args,out=None):
        if module in watched:seen.append((id(module),(args[0] if pre else out).clone()))
    if target=='global':
        register=torch.nn.modules.module.register_module_forward_pre_hook if pre else torch.nn.modules.module.register_module_forward_hook
        handle=register(hook)
    else:
        module=getattr(layer,target);handle=(module.register_forward_pre_hook if pre else module.register_forward_hook)(hook)
    try:
        ref=layer._ff_block(x);expected=seen.copy();seen.clear()
        with optimized(model,**OPTIONS):
            bits(layer._ff_block(x),ref);assert model._fast_matrix_runtime.residual_async_calls==0
            assert len(seen)==len(expected)
            for (i,a),(j,b) in zip(seen,expected):assert i==j;bits(a,b)
    finally:handle.remove()

@torch.inference_mode()
@pytest.mark.parametrize('target',['linear1','linear2','norm2','activation'])
def test_residual_async_custom_paths(monkeypatch,target):
    model,layer,x=fixture(monkeypatch)
    if target=='activation':layer.activation=F.relu
    else:
        module=getattr(layer,target);original=module.forward;module.forward=lambda value:original(value)+.01
    ref=layer._ff_block(x)
    with optimized(model,**OPTIONS):
        bits(layer._ff_block(x),ref);assert model._fast_matrix_runtime.residual_async_calls==0


def test_residual_async_gradient_and_autocast_fallback(monkeypatch):
    model,layer,x=fixture(monkeypatch);original=layer._ff_block
    with optimized(model,**OPTIONS):
        z=x.detach().requires_grad_(True);a=layer._ff_block(z);b=original(z)
        bits(a,b);bits(torch.autograd.grad(a.sum(),z)[0],torch.autograd.grad(b.sum(),z)[0])
        with torch.autocast('cuda'):assert torch.equal(layer._ff_block(x),original(x))
        assert model._fast_matrix_runtime.residual_async_calls==0

@torch.inference_mode()
def test_residual_async_remains_cuda_only(monkeypatch):
    model,layer,x=fixture(monkeypatch);ref=layer._ff_block(x)
    with optimized(model,**dict(OPTIONS,matrix_backend='triton',norm_backend='none')):
        bits(layer._ff_block(x),ref);assert model._fast_matrix_runtime.residual_async_calls==0

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from fast_moss.norm_async import CONFIGS,SOURCE,linear,compile_kernel
from fast_moss.norm_projection import project
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformerLayer

OPTIONS=dict(matrix_backend='cuda',norm_backend='cuda',ffn_backend='triton',residual_backend='triton',attention_mask_backend='triton')

def fixture(monkeypatch):
    strict_precision();torch.manual_seed(1001)
    layer=MossAudioTokenizerTransformerLayer(1280,20,dim_feedforward=5120,layer_scale=.1)
    model=torch.nn.Sequential(layer).cuda().eval().requires_grad_(False)
    profile=json.loads(Path('fast_moss/matrix_profile.json').read_text());monkeypatch.setattr('fast_moss.matrices.load_profile',lambda m:profile)
    return model,layer,torch.randn(1,1,1280,device='cuda')

def bits(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.inference_mode()
@pytest.mark.parametrize('mode',['none','gelu'])
def test_norm_async_production_kernel_matches_research(mode):
    from benchmarks.norm_gemv_async import SOURCE as research_source
    assert SOURCE==research_source
    strict_precision();torch.manual_seed(1002)
    n=3840 if mode=='none' else 5120
    x=torch.randn(1,1280,device='cuda');w=torch.randn(n,1280,device='cuda')*.03
    g=torch.randn(1280,device='cuda');b=torch.randn_like(g)*.01
    fn=lambda a,ww:linear(a,ww,g,b,1e-5,mode,debug=True)
    graph=GraphedCallable(fn,x,w)
    for a,ww in [(x,w),(-x,w),(x*1e-38,w),(x*1e10,w),(x*.001+10000,w),
                 (torch.full_like(x,-0.),w),(x,torch.full_like(w,1.401298464324817e-45)),(x,w)]:
        norm=F.layer_norm(a,(1280,),g,b,1e-5);ref=F.linear(norm,ww)
        if mode=='gelu':ref=F.gelu(ref)
        for y,z in [fn(a,ww),graph(a,ww)]:bits(y,ref);bits(z,norm)
    _,_,res,_=compile_kernel(n,mode,CONFIGS[mode])
    prior=json.loads(Path('results/norm_gemv_async_ring_confirm.json').read_text())
    assert res==next(r['resources'][json.dumps(CONFIGS[mode])] for r in prior['records'] if r['mode']==mode)

@torch.inference_mode()
@pytest.mark.parametrize('block',['_sa_block','_ff_block'])
@pytest.mark.parametrize('backend',['triton','cute'])
def test_norm_async_switch_capture_and_restore(monkeypatch,block,backend):
    model,layer,x=fixture(monkeypatch);original=getattr(layer,block);ref=original(x)
    pointers=[p.data_ptr() for p in model.parameters()]
    with optimized(model,**dict(OPTIONS,residual_backend=backend)):
        owner=model._fast_matrix_runtime;fn=getattr(layer,block);bits(fn(x),ref)
        assert owner.norm_async_calls==1 and owner.norm_gemv_calls==1
        assert (owner.norm_async_qkv_calls,owner.norm_async_ffn_calls)==((1,0) if block=='_sa_block' else (0,1))
        staged=GraphedCallable(lambda z:(fn(z),),x);before=owner.norm_async_calls
        owner.norm_async_enabled=False;bits(fn(x),ref);assert owner.norm_async_calls==before
        direct=GraphedCallable(lambda z:(fn(z),),x)
        for z in (x,-x,x*.001+10000):
            bits(staged(z)[0],original(z));bits(direct(z)[0],original(z))
        owner.norm_async_enabled=True;bits(fn(x),ref);assert owner.norm_async_calls==before+1
        assert [p.data_ptr() for p in model.parameters()]==pointers and owner.packed_bytes==0
    assert not owner.norm_gemv_warmed and not owner.forwards
    assert getattr(layer,block)==original
    assert [p.data_ptr() for p in model.parameters()]==pointers
    for graph in (staged,direct):
        with pytest.raises(RuntimeError,match='storage changed'):graph(x)

@torch.inference_mode()
@pytest.mark.parametrize('first_async',[False,True])
@pytest.mark.parametrize('mode',['none','gelu'])
def test_norm_async_warmup_is_per_backend_and_stream(monkeypatch,first_async,mode):
    model,layer,x=fixture(monkeypatch)
    norm,projection=(layer.norm1,layer.self_attn.in_projs[0]) if mode=='none' else (layer.norm2,layer.linear1)
    with optimized(model,**OPTIONS):
        owner=model._fast_matrix_runtime
        def run():return project(owner,x,norm,projection,mode)
        owner.norm_async_enabled=first_async;assert run() is not None
        owner.norm_async_enabled=not first_async
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm normalization/projection'):run()
        assert run() is not None
        parent=torch.cuda.current_stream();side=torch.cuda.Stream();side.wait_stream(parent)
        with torch.cuda.stream(side):
            for backend in (False,True):
                owner.norm_async_enabled=backend
                with monkeypatch.context() as patch:
                    patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
                    with pytest.raises(RuntimeError,match='Warm normalization/projection'):run()
                assert run() is not None
        parent.wait_stream(side);torch.cuda.synchronize()
        assert len(owner.norm_gemv_warmed)==4 and owner.norm_async_calls==2
    assert not owner.norm_gemv_warmed

@torch.inference_mode()
@pytest.mark.parametrize('mode',['none','gelu'])
def test_norm_async_packing_and_exception_cleanup(monkeypatch,mode):
    model,layer,x=fixture(monkeypatch)
    norm,projection=(layer.norm1,layer.self_attn.in_projs[0]) if mode=='none' else (layer.norm2,layer.linear1)
    original=projection.forward;weight=projection.weight.clone()
    with pytest.raises(RuntimeError,match='body failure'):
        with optimized(model,**OPTIONS):
            owner=model._fast_matrix_runtime
            graph=GraphedCallable(lambda z:(project(owner,z,norm,projection,mode),),x)
            before=owner.norm_async_calls
            projection(torch.randn(1,24,1280,device='cuda'));assert not projection.weight.is_contiguous()
            assert project(owner,x,norm,projection,mode) is None and owner.norm_async_calls==before
            with pytest.raises(RuntimeError,match='storage changed'):graph(x)
            raise RuntimeError('body failure')
    bits(projection.weight,weight);assert projection.weight.is_contiguous() and projection.forward==original
    assert not owner.norm_gemv_warmed and not hasattr(model,'_fast_matrix_runtime')

@torch.inference_mode()
def test_norm_async_operand_guards():
    x=torch.zeros(1,1280,device='cuda');w=torch.zeros(3840,1280,device='cuda');g=torch.ones(1280,device='cuda');b=torch.zeros_like(g)
    for mode,weight in [('invalid',w),('gelu',w),('none',w[:,1:])]:
        with pytest.raises(ValueError,match='aligned'):linear(x,weight,g,b,1e-5,mode)
    shifted=torch.empty(w.numel()+1,device='cuda')[1:].reshape_as(w)
    with pytest.raises(ValueError,match='aligned'):linear(x,shifted,g,b,1e-5,'none')

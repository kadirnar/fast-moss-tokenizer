import pytest
import torch
import torch.nn.functional as F
from benchmarks.fixtures import structural_model
from fast_moss.loading import strict_precision
from fast_moss.optimize import optimized
from fast_moss.projections import project, reconstruct
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
@pytest.mark.parametrize('batch,frames', [(1,1),(1,3),(8,3),(128,3),(8,40)])
def test_project_matches_cudnn_order_layout_bias_and_graph(batch, frames):
    strict_precision()
    torch.manual_seed(831)
    w=torch.randn(512,8,1,device='cuda')*.1
    bias=torch.randn(512,device='cuda')*.1
    for scale in [0.,.2,1e-38,1e10]:
        # A B,T,C allocation has the decoder embedding's strided B,C,T view.
        x=(torch.randn(batch,frames,8,device='cuda')*scale).transpose(1,2)
        ref=F.conv1d(x,w,bias)
        out=project(x,w,bias)
        assert out.stride()==ref.stride() and torch.equal(out,ref)
        graph=GraphedCallable(lambda z:(project(z,w,bias),),x)
        assert torch.equal(graph(x)[0],ref)
    x=torch.randn(batch,8,frames,device='cuda')
    assert torch.equal(project(x,w,None),F.conv1d(x,w))


@torch.inference_mode()
def test_projected_cache_decoder_all_codes_counts_strides_and_restoration():
    model=structural_model()
    codes=torch.arange(1024,device='cuda')[None,None].expand(32,1,-1).contiguous()
    cases=[codes,codes.reshape(32,1024,1),codes[:7,:,::3],codes[:0,:,:3],codes[:1,:,:3],
           torch.cat([codes[:,:,:3],torch.full((1,1,3),-1,device='cuda')]),codes[:1,:,:3].int()]
    refs=[model.quantizer.decode_codes(c) for c in cases]
    with optimized(model,projection_backend='triton'):
        assert model.quantizer._fast_projected_codebooks.numel()*4==64*1024*1024
        for c,ref in zip(cases,refs):
            assert torch.equal(model.quantizer.decode_codes(c),ref)
            graph=GraphedCallable(lambda z:(model.quantizer.decode_codes(z),),c)
            assert torch.equal(graph(c)[0],ref)
    with pytest.raises(RuntimeError,match='cached resources expired'):
        graph(cases[-1])
    assert '_fast_projected_codebooks' not in model.quantizer.__dict__
    assert 'decode_codes' not in model.quantizer.__dict__
    assert all('forward' not in q.out_proj.__dict__ for q in model.quantizer.quantizers)
    assert torch.equal(model.quantizer.decode_codes(cases[0]),refs[0])


@torch.inference_mode()
def test_projection_hooks_fallback_autocast_and_cleanup():
    model=structural_model()
    codes=torch.randint(1024,(32,2,3),device='cuda')
    ref=model.quantizer.decode_codes(codes)
    calls=[]
    handle=model.quantizer.quantizers[0].out_proj.register_forward_hook(lambda *args:calls.append(1))
    with optimized(model,projection_backend='triton'):
        assert calls==[]  # Cache setup must not emit observer calls.
        assert torch.equal(model.quantizer.decode_codes(codes),ref)
        assert calls==[1]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            old=model.quantizer._fast_original_decode_codes(codes)
            assert torch.equal(model.quantizer.decode_codes(codes),old)
    handle.remove()
    with pytest.raises(RuntimeError,match='injected'):
        with optimized(model,projection_backend='triton'):
            raise RuntimeError('injected')
    assert '_fast_projected_codebooks' not in model.quantizer.__dict__
    assert '_fast_optimization_active' not in model.__dict__


def test_projection_rejects_invalid_options(monkeypatch):
    model=structural_model()
    for kwargs in [dict(projection_backend='bad'),dict(projection_backend='triton',cache_weights=False)]:
        with pytest.raises(ValueError):
            with optimized(model,**kwargs):pass
    with pytest.raises(ValueError):project(torch.zeros(1,8,1),torch.zeros(512,8,1),None)
    monkeypatch.setattr(torch.backends.cudnn,'version',lambda:0)
    with pytest.raises(ValueError,match='validated'):
        with optimized(model,projection_backend='triton'):pass
    assert '_fast_optimization_active' not in model.__dict__


@torch.inference_mode()
@pytest.mark.parametrize('batch,frames',[(1,1),(8,3),(8,40)])
def test_projection_update_preserves_each_masked_rounding(batch,frames):
    from fast_moss.projections import project_update
    strict_precision(); torch.manual_seed(754)
    x=torch.randn(batch,8,frames,device='cuda')*.2
    w=torch.randn(512,8,1,device='cuda')*.1
    bias=torch.randn(512,device='cuda')*.1
    residual=torch.randn(batch,512,frames,device='cuda')*.02
    mask=torch.rand(batch,frames,device='cuda')>.4
    expected=residual-F.conv1d(x,w,bias)*mask[:,None]
    actual,masked=project_update(x,w,bias,residual,mask)
    assert torch.equal(actual,expected) and torch.equal(masked,expected*mask[:,None])
    graph=GraphedCallable(lambda a,b:project_update(a,w,bias,b,mask),x,residual)
    actual,masked=graph(x,residual)
    assert torch.equal(actual,expected) and torch.equal(masked,expected*mask[:,None])


def test_invalid_decoder_codes_raise_in_isolated_cuda_context():
    # Native CUDA embedding assertions also invalidate their CUDA context. Test
    # the asynchronous rejection in an isolated process, not the test runner.
    import subprocess,sys
    script='''
import torch
from fast_moss.projections import reconstruct
codes=torch.full((1,1,1),1024,device='cuda',dtype=torch.long)
table=torch.zeros(32,1024,512,device='cuda')
reconstruct(codes,table)
torch.cuda.synchronize()
'''
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Invalid decoder code index' in result.stderr or 'device-side assert' in result.stderr


@torch.inference_mode()
@pytest.mark.parametrize('frames',[1,3])
def test_full_structural_projection_encoder_decoder_and_observers(frames):
    model=structural_model()
    x=torch.randn(2,1,frames*1920,device='cuda')*.05
    def run(value):
        enc=model._encode_frame(value)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    refs=run(x)
    with optimized(model,residual_backend='triton',kv_backend='triton',rope_backend='triton',
                   share_rope_tables=True,attention_mask_backend='triton',quantizer_backend='triton',
                   projection_backend='triton'):
        assert all(torch.equal(a,b) for a,b in zip(refs,run(x)))
        graph=GraphedCallable(run,x)
        assert all(torch.equal(a,b) for a,b in zip(refs,graph(x)))
        calls=[]
        hook=model.quantizer.quantizers[0].out_proj.register_forward_hook(lambda *args:calls.append(1))
        assert all(torch.equal(a,b) for a,b in zip(refs,run(x)))
        assert len(calls)==2  # Encoder and decoder preserve observed projection calls.
        hook.remove()


def test_projection_cache_setup_failure_restores_partial_changes(monkeypatch):
    model=structural_model()
    original=F.conv1d
    count=0
    def fail(*args,**kwargs):
        nonlocal count
        count+=1
        if count==3:raise RuntimeError('injected cache setup failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(F,'conv1d',fail)
    with pytest.raises(RuntimeError,match='injected'):
        with optimized(model,projection_backend='triton'):pass
    assert '_fast_optimization_active' not in model.__dict__
    assert '_fast_projected_codebooks' not in model.quantizer.__dict__
    for q in model.quantizer.quantizers:
        assert '_fast_original_projection' not in q.out_proj.__dict__
        assert '_fast_weight' not in q.out_proj.__dict__
        assert 'forward' not in q.out_proj.__dict__

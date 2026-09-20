"""Reduced v2 architecture tests; full checkpoint evidence lives in benchmarks.v2_* ."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from transformers import AutoConfig, AutoModel

from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision
from fast_moss.v2 import V2Runtime, codec, encode_fixed, optimized
from fast_moss.v2_loading import MODEL_ID, REVISION


@pytest.fixture
def model():
    strict_precision()
    with torch.inference_mode(False):
        torch.manual_seed(24048)
        config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
        config.attention_implementation = 'sdpa'
        config.compute_dtype = 'bf16'
        config.codec_weight_dtype = 'fp32'
        for group in (config.encoder_kwargs, config.decoder_kwargs):
            for block in group:
                if block['module_type'] == 'Transformer':
                    block.update(num_layers=1, d_model=64, dim_feedforward=128, num_heads=1)
        return AutoModel.from_config(config, trust_remote_code=True).eval().cuda().requires_grad_(False)


def equal(a, b):
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.equal(x.view(torch.int32) if x.dtype == torch.float32 else x,
                           y.view(torch.int32) if y.dtype == torch.float32 else y)


def native(model, x):
    enc = model._encode_frame(x)
    return enc.audio_codes, enc.encoder_hidden_states, model._decode_frame(enc.audio_codes).audio


@torch.inference_mode()
@pytest.mark.parametrize('batch,frames', [(1,1), (2,1), (1,3)])
def test_v2_stereo_reference_graph_and_restore(model, batch, frames):
    x = torch.randn(batch,2,frames*3840,device='cuda')*.1
    changed = x.flip(1).contiguous()
    ref, changed_ref = native(model,x), native(model,changed)
    pointers = [p.data_ptr() for p in model.parameters()]
    versions = [p._version for p in model.parameters()]
    with optimized(model) as owner:
        equal(ref,native(model,x))
        graph = GraphedCallable(lambda z:codec(model,z),x)
        equal(ref,graph(x));equal(changed_ref,graph(changed));equal(ref,graph(x))
        assert owner.linear_calls and owner.prepare_calls and owner.select_calls
        assert owner.conv_modules == 66 and owner.quantizer_modules == 32
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert [p.data_ptr() for p in model.parameters()] == pointers
        assert [p._version for p in model.parameters()] == versions
    equal(ref,native(model,x))
    with pytest.raises(RuntimeError,match='storage changed'):
        graph(x)


def test_v2_linear_cache_autocast_gradient_mutation_and_thread_guards(model):
    layer = next(m for m in model.modules() if type(m) is torch.nn.Linear)
    original = layer.forward
    x = torch.randn(2,layer.in_features,device='cuda')
    with optimized(model) as owner:
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            equal((original(x),),(layer(x),))
            assert owner.linear_calls == 1
        equal((original(x),),(layer(x),))
        assert owner.linear_calls == 1
        with torch.autocast('cuda',dtype=torch.bfloat16):
            x_grad = x.clone().requires_grad_(True)
            expected = torch.autograd.grad(original(x_grad).float().sum(),x_grad)[0]
            actual = torch.autograd.grad(layer(x_grad).float().sum(),x_grad)[0]
            equal((expected,),(actual,))
        assert owner.linear_calls == 1
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            layer.weight.add_(.01)
            equal((original(x),),(layer(x),))
        assert owner.linear_calls == 1
        with ThreadPoolExecutor(1) as pool:
            with pytest.raises(RuntimeError,match='owner thread'):
                pool.submit(layer,x).result()
    assert layer.forward == original


@torch.inference_mode()
def test_v2_conv_parametrization_hooks_and_mutation(model):
    layer = model.quantizer.quantizers[0].in_proj
    original = layer.forward
    x = torch.randn(1,layer.in_channels,3,device='cuda')
    with optimized(model):
        equal((original(x),),(layer(x),))
        calls = []
        hook = layer.parametrizations.weight.register_forward_hook(lambda *args:calls.append(1))
        try:
            equal((original(x),),(layer(x),))
            assert len(calls) == 2
        finally:
            hook.remove()
        next(layer.parametrizations.weight.parameters()).add_(.01)
        equal((original(x),),(layer(x),))


@torch.inference_mode()
def test_v2_quantizer_layout_mutation_and_cold_capture_guards(model, monkeypatch):
    q = model.quantizer.quantizers[0]
    original = q.decode_latents
    x = torch.randn(2,8,6,device='cuda')
    with optimized(model) as owner:
        equal(original(x[...,::2]),q.decode_latents(x[...,::2]))
        assert owner.prepare_calls == 0
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda,'is_current_stream_capturing',lambda:True)
            with pytest.raises(RuntimeError,match='Warm v2 quantizer'):
                q.decode_latents(x)
        equal(original(x),q.decode_latents(x))
        assert owner.prepare_calls == 1
        q.codebook.weight.add_(.1)
        equal(original(x),q.decode_latents(x))
        assert owner.prepare_calls == 1


def test_v2_context_exception_reuse_and_validation(model):
    original = {id(m):m.forward for m in model.modules()}
    owner = V2Runtime(model)
    with pytest.raises(ValueError,match='test failure'):
        with owner:
            with pytest.raises(RuntimeError,match='already active'):
                with optimized(model):
                    pass
            raise ValueError('test failure')
    assert not owner.active and not owner.saved and not owner.warmed
    assert not hasattr(model,'_fast_v2_runtime')
    assert all(m.forward == original[id(m)] for m in model.modules())
    with pytest.raises(RuntimeError,match='has been used'):
        with owner:
            pass
    model.config._commit_hash = 'different checkpoint'
    with pytest.raises(ValueError,match='pinned'):
        V2Runtime(model)


@pytest.mark.parametrize('shape', [(1,1,3840),(1,2,1920),(0,2,3840),(1,2,4000)])
def test_v2_fixed_adapter_rejects_incomplete_or_mono_input(model, shape):
    with pytest.raises(ValueError,match='complete equal-length'):
        encode_fixed(model,torch.zeros(shape,device='cuda'))

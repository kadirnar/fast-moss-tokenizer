import math
import pytest
import torch
from fast_moss.rope import rotate
from upstream.modeling_moss_audio_tokenizer import apply_rope
from benchmarks.fixtures import structural_model
from fast_moss.optimize import optimized


@torch.inference_mode()
@pytest.mark.parametrize("shape",[(1,20,1,64),(2,12,8,64),(3,2,5,8)])
@pytest.mark.parametrize("offset_value",[0,1234567,2**27])
def test_rotary_bitwise_with_packed_strides(shape,offset_value):
    b,h,t,d=shape
    torch.manual_seed(303)
    packed=torch.randn(b,t,3,h,d,device="cuda").permute(2,0,3,1,4)
    q,k=packed[0],packed[1]
    offset=torch.arange(b,device="cuda",dtype=torch.long)+offset_value
    freq=torch.exp(torch.arange(d//2,device="cuda",dtype=torch.float32)*(-math.log(10000)*2/d))
    ts=offset.float().view(-1,1)+torch.arange(t,device="cuda",dtype=torch.float32)
    phase=freq*ts.view(b,t,1)
    actual=rotate(q,k,torch.cos(phase),torch.sin(phase))
    expected=apply_rope(q,k,offset)
    assert all(torch.equal(a,e) for a,e in zip(actual,expected))


@torch.inference_mode()
def test_rotary_integration_exact():
    model=structural_model()
    x=torch.randn(2,1,5760,device="cuda")*.05
    ref=model.encode(x,return_dict=True)
    ref_audio=model.decode(ref.audio_codes,return_dict=True).audio
    with optimized(model,rope_backend="triton"):
        actual=model.encode(x,return_dict=True)
        assert torch.equal(actual.audio_codes,ref.audio_codes)
        assert torch.equal(actual.encoder_hidden_states,ref.encoder_hidden_states)
        assert torch.equal(model.decode(actual.audio_codes,return_dict=True).audio,ref_audio)


@torch.inference_mode()
@pytest.mark.parametrize("mask_backend", ["none", "triton"])
def test_shared_tables_multiple_layers_and_exception_cleanup(mask_backend):
    from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformer
    from fast_moss.graphs import GraphedCallable
    stage=MossAudioTokenizerTransformer(768,12,num_layers=3,dim_feedforward=3072,
                                        causal=True,context=16,positional_embedding="rope").cuda().eval().requires_grad_(False)
    x=torch.randn(2,8,768,device="cuda")
    ref=stage(x)
    with optimized(stage,rope_backend="triton",share_rope_tables=True,
                   attention_mask_backend=mask_backend):
        assert torch.equal(stage(x),ref)
        graph=GraphedCallable(lambda z:(stage(z),),x)
        assert torch.equal(graph(x)[0],ref)
        assert stage.rope._fast_tables is None
        assert all(getattr(layer.self_attn, "_fast_mask_pool", None) is None for layer in stage.layers)
        def fail(*args):
            raise RuntimeError("injected")
        hook=stage.layers[1].register_forward_pre_hook(fail)
        try:
            with pytest.raises(RuntimeError,match="injected"):
                stage(x)
            assert stage.rope._fast_tables is None
            assert all(getattr(layer.self_attn, "_fast_mask_pool", None) is None for layer in stage.layers)
        finally:
            hook.remove()


@torch.inference_mode()
@pytest.mark.parametrize("mask_backend", ["none", "triton"])
def test_unsynchronized_streaming_falls_back(mask_backend):
    from upstream.modeling_moss_audio_tokenizer import MossAudioTokenizerTransformer
    stage=MossAudioTokenizerTransformer(64,1,num_layers=2,dim_feedforward=128,
                                        causal=True,context=16,positional_embedding="rope").cuda().eval().requires_grad_(False)
    x=torch.randn(1,2,64,device="cuda")
    def run():
        with stage.streaming(1):
            # An external caller can give layers different execution histories.
            stage.layers[1].self_attn._streaming_state.offset.fill_(2)
            return stage(x)
    reference=run()
    with optimized(stage,rope_backend="triton",share_rope_tables=True,
                   attention_mask_backend=mask_backend):
        assert torch.equal(run(),reference)

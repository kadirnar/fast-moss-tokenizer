"""Fast structural tests; these do not replace full-checkpoint fidelity validation."""
import pytest
import torch

from fast_moss.graphs import GraphedCallable
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession
from benchmarks.fixtures import structural_model


@pytest.fixture(scope="module")
def model():
    return structural_model()


@torch.inference_mode()
@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_model_optimization_exact(model, backend):
    x = torch.randn(2, 1, 3840, device="cuda") * .05
    reference = model.encode(x, return_dict=True)
    audio = model.decode(reference.audio_codes, return_dict=True).audio
    with optimized(model, residual_backend=backend):
        candidate = model.encode(x, return_dict=True)
        assert torch.equal(reference.audio_codes, candidate.audio_codes)
        assert torch.equal(reference.encoder_hidden_states, candidate.encoder_hidden_states)
        assert torch.equal(audio, model.decode(candidate.audio_codes, return_dict=True).audio)
    assert torch.equal(audio, model.decode(reference.audio_codes, return_dict=True).audio)


@torch.inference_mode()
@pytest.mark.parametrize("direction", ["encode", "decode"])
@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_streaming_graph_wrap_reset_batch_and_tail(model, direction, backend):
    torch.manual_seed(19)
    shape = (2, 1, 1920) if direction == "encode" else (32, 2, 1)
    chunks = [(torch.randn(shape, device="cuda") * .05 if direction == "encode"
               else torch.randint(1024, shape, device="cuda")) for _ in range(5)]
    with optimized(model, residual_backend=backend, kv_backend="triton"):
        with StreamingSession(model, direction, batch_size=2, use_graph=False,fast_reset=False) as session:
            reference = [session.push(x)[0].clone() for x in chunks]
        with StreamingSession(model, direction, batch_size=2) as session:
            candidate = [session.push(x)[0] for x in chunks]
            for ref, actual in zip(reference, candidate):
                torch.testing.assert_close(actual, ref, rtol=0, atol=0)
            session.reset()
            assert torch.equal(session.push(chunks[0])[0], reference[0])
            # Outputs from older replays must not alias the graph buffer.
            assert torch.equal(candidate[0], reference[0])
            if direction == "encode":
                tail, lengths = session.push(chunks[0][..., :555], final=True)
                assert tail.shape[-1] == 1
                assert lengths.tolist() == [1, 1]
            else:
                session.push(chunks[0], final=True)
            with pytest.raises(RuntimeError):
                session.push(chunks[0])


@torch.inference_mode()
def test_graph_input_validation_and_ownership():
    x = torch.ones(64, device="cuda")
    graph = GraphedCallable(lambda a: (a * 2,), x)
    first = graph(x)[0]
    second = graph(x + 1)[0]
    assert first.eq(2).all() and second.eq(4).all()
    with pytest.raises(ValueError):
        graph(x[:3])


def test_session_cleanup(model):
    with pytest.raises(RuntimeError, match="test cleanup"):
        with StreamingSession(model, "encode", use_graph=False):
            with pytest.raises(RuntimeError, match="already"):
                with StreamingSession(model, "decode", use_graph=False):
                    pass
            raise RuntimeError("test cleanup")
    assert not any(getattr(m, "_streaming_state", None) is not None for m in model.modules())
    assert not hasattr(model, "_fast_streaming_owner")


def test_nested_optimization_is_rejected_and_restored(model):
    original=model.quantizer.input_proj.forward
    with optimized(model):
        with pytest.raises(RuntimeError, match="active optimization"):
            with optimized(model):
                pass
        assert model._fast_optimization_active
    assert not hasattr(model,"_fast_optimization_active")
    assert model.quantizer.input_proj.forward==original

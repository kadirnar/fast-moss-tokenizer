import pytest
import torch
from benchmarks.fixtures import structural_model
from fast_moss.batching import StreamingBatcher
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession


@torch.inference_mode()
def independent(model, direction, inputs, batch=3, frames=2):
    """Original eager arithmetic on independent timelines at the same batch size."""
    outputs = []
    width = frames * (model.downsample_rate if direction == 'encode' else 1)
    for value in inputs:
        chunks = []
        with StreamingSession(model, direction, batch, frames, use_graph=False, fast_reset=False) as s:
            for start in range(0, value.shape[-1], width):
                part = value[..., start:start + width]
                x = (part[None].expand(batch, -1, -1).contiguous() if direction == 'encode'
                     else part[:, None].expand(-1, batch, -1).contiguous())
                out, _ = s.push(x, final=start + width >= value.shape[-1])
                chunks.append((out[:, 0] if direction == 'encode' else out[0]).clone())
        outputs.append(torch.cat(chunks, -1))
    return outputs


@torch.inference_mode()
@pytest.mark.parametrize('direction', ['encode', 'decode'])
@pytest.mark.parametrize('graph', [False, True])
def test_queued_requests_refill_and_preserve_timelines(direction, graph):
    model = structural_model()
    torch.manual_seed(135)
    sizes = [9, 1, 2, 3, 7, 4, 1]
    inputs = [torch.randn(1, n * 1920 - 17, device='cuda') * .05 if direction == 'encode'
              else torch.randint(1024, (32, n), device='cuda') for n in sizes]
    refs = independent(model, direction, inputs)
    outputs = {i: [] for i in range(len(inputs))}
    positions = [0] * len(inputs)
    finals = []; lanes = {}
    with optimized(model, residual_backend='triton', kv_backend='triton', rope_backend='triton',
                   share_rope_tables=True, attention_mask_backend='triton'):
        with StreamingBatcher(model, direction, 3, 2, use_graph=graph) as batcher:
            for x in inputs[:4]: batcher.submit(x)
            first = True; saved_graph = None
            while batcher.pending:
                for chunk in batcher.step():
                    i = chunk.request_id
                    assert chunk.offset == positions[i]
                    positions[i] += chunk.data.shape[-1]
                    assert lanes.setdefault(i, chunk.lane) == chunk.lane
                    outputs[i].append(chunk.data)
                    if chunk.final: finals.append(i)
                if first:
                    saved_graph = batcher.session.graph
                    for x in inputs[4:]: batcher.submit(x)
                    first = False
                assert batcher.session.graph is saved_graph
            assert batcher.step() == []
    assert sorted(finals) == list(range(len(inputs)))
    for i, ref in enumerate(refs):
        assert torch.equal(torch.cat(outputs[i], -1), ref)


@torch.inference_mode()
def test_cancellation_ownership_capacity_empty_and_lifecycle():
    model = structural_model()
    x = torch.randn(1, 1920, device='cuda') * .05
    ref = independent(model, 'encode', [x], batch=2, frames=1)[0]
    with optimized(model, residual_backend='triton', kv_backend='triton'):
        with StreamingBatcher(model, 'encode', 2, 1, max_pending=3) as batcher:
            long_id = batcher.submit(x.repeat(1, 4))
            batcher.submit(x.repeat(1, 4))
            queued = batcher.submit(x)
            with pytest.raises(BufferError): batcher.submit(x)
            batcher.cancel(queued)
            with pytest.raises(KeyError): batcher.cancel(queued)
            batcher.step()
            graph = batcher.session.graph
            batcher.cancel(long_id)
            owned_id = batcher.submit(x)
            x.fill_(float('nan'))  # Submission owns its data.
            out = batcher.step()
            assert torch.equal(next(c.data for c in out if c.request_id == owned_id), ref)
            empty_id = batcher.submit(x[..., :0])
            out = batcher.step()
            empty = next(c for c in out if c.request_id == empty_id)
            assert empty.final and empty.data.shape == (32, 0) and empty.input_length == 0
            assert batcher.session.graph is graph
            other = torch.cuda.Stream()
            with torch.cuda.stream(other):
                with pytest.raises(RuntimeError, match='construction stream'): batcher.step()
                with pytest.raises(RuntimeError, match='construction stream'): batcher.submit(x)
                with pytest.raises(RuntimeError, match='construction stream'): batcher.cancel(1)
        assert batcher.pending == 0
        with pytest.raises(RuntimeError): batcher.step()
        with pytest.raises(RuntimeError): batcher.__enter__()
    assert getattr(model, '_fast_streaming_owner', None) is None


@torch.inference_mode()
def test_batcher_rejects_invalid_configuration_and_storage():
    model = structural_model()
    for kwargs in [{'batch_size': 0}, {'chunk_frames': 1.5}, {'max_pending': True}]:
        with pytest.raises(ValueError): StreamingBatcher(model, 'encode', **kwargs)
    with pytest.raises(ValueError, match='kv_backend'):
        with StreamingBatcher(model, 'encode', chunk_frames=1): pass
    assert getattr(model, '_fast_streaming_owner', None) is None
    with optimized(model, kv_backend='triton'):
        with StreamingBatcher(model, 'encode', 2, 1) as batcher:
            for x in [torch.zeros(1, 1920), torch.zeros(2, 1920, device='cuda'),
                      torch.zeros(1, 1920, device='cuda', dtype=torch.float64)]:
                with pytest.raises(ValueError): batcher.submit(x)
            batcher.submit(torch.zeros(1, 0, device='cuda'))
            assert batcher.step()[0].final
            assert batcher.session.graph is None

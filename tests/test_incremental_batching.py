import pytest
import torch
from benchmarks.fixtures import structural_model
from fast_moss.batching import StreamingBatcher, _gather_chunks
from fast_moss.optimize import optimized
from tests.test_batching import independent


@torch.inference_mode()
@pytest.mark.parametrize('direction', ['encode', 'decode'])
def test_segmented_gather_preserves_bits_padding_and_nonzero_offsets(direction):
    encode = direction == 'encode'
    channels, width = (1, 1153) if encode else (32, 19)
    # Arbitrary bit patterns include NaNs; gathering must not do arithmetic.
    dtype = torch.int32 if encode else torch.int64
    raw = [torch.randint(-(2**30), 2**30, (channels, width * 3), device='cuda', dtype=dtype) for _ in range(5)]
    values = [x.view(torch.float32) if encode else x for x in raw]
    cut = width // 3
    segments = [[(values[0], 3, cut), (values[1], 7, width-cut)], [],
                [(values[2], 1, 1), (values[3], 4, cut), (values[4], 2, 2)]]
    out = _gather_chunks(segments, width=width, channels=channels, encode=encode, device='cuda')
    expected = torch.zeros_like(out)
    for lane, parts in enumerate(segments):
        offset = 0
        for value, source, n in parts:
            target = expected[lane] if encode else expected[:, lane]
            target[..., offset:offset+n].copy_(value[..., source:source+n])
            offset += n
    assert torch.equal(out.view(dtype), expected.view(dtype))


@torch.inference_mode()
@pytest.mark.parametrize('direction', ['encode', 'decode'])
@pytest.mark.parametrize('use_graph', [False, True])
def test_incremental_uneven_arrivals_pause_final_and_lane_reuse(direction, use_graph):
    model = structural_model()
    width = 3840 if direction == 'encode' else 2
    channels = 1 if direction == 'encode' else 32
    torch.manual_seed(1073)
    def source(length):
        return (torch.randn(channels, length, device='cuda') * .05 if direction == 'encode'
                else torch.randint(1024, (channels, length), device='cuda'))
    inputs = [source(width * 5 + 1), source(width * 2), source(width + 1), source(width)]
    refs = independent(model, direction, inputs, batch=2, frames=2)
    results = {i: [] for i in range(4)}; finals = []; lanes = {}; positions = [0] * 4
    def collect(chunks):
        for c in chunks:
            assert c.offset == positions[c.request_id]
            positions[c.request_id] += c.data.shape[-1]
            results[c.request_id].append(c.data)
            assert lanes.setdefault(c.request_id, c.lane) == c.lane
            if c.final: finals.append(c.request_id)
    with optimized(model, residual_backend='triton', kv_backend='triton', rope_backend='triton',
                   share_rope_tables=True, attention_mask_backend='triton', quantizer_backend='triton'):
        with StreamingBatcher(model, direction, 2, 2, use_graph=use_graph) as queue:
            assert queue.submit(inputs[0][..., :1], final=False) == 0
            assert not queue.can_step and queue.step() == []
            assert queue.session.graph is None
            # A ready request enters ahead of an older incomplete queued request.
            assert queue.submit(inputs[1][..., :width], final=False) == 1
            collect(queue.step())
            assert lanes[1] == 0 and 0 not in lanes
            captured = queue.session.graph
            offsets = [m._streaming_state.offset.clone() for m in model.modules()
                       if getattr(getattr(m, '_streaming_state', None), 'offset', None) is not None]
            assert not queue.can_step and queue.step() == []
            current = [m._streaming_state.offset for m in model.modules()
                       if getattr(getattr(m, '_streaming_state', None), 'offset', None) is not None]
            assert all(torch.equal(x, y) for x, y in zip(offsets, current))
            # Assemble request 0's first chunk across three owned fragments.
            fragment = inputs[0][..., 1:width//2].clone()
            queue.append(0, fragment)
            fragment.zero_()  # append owns a copy even while its request waits.
            queue.append(0, inputs[0][..., width//2:width])
            collect(queue.step())
            assert lanes[0] == 1
            queue.append(1, inputs[1][..., width:])
            collect(queue.step())
            queue.append(1, final=True)  # Final notification after its last full output.
            collect(queue.step())
            assert 1 in finals and queue.pending == 1
            assert queue.submit(inputs[2]) == 2
            queue.append(0, inputs[0][..., width:2*width])
            collect(queue.step())
            collect(queue.step())  # Request 2 finishes while 0 waits for more input.
            assert lanes[2] == 0 and 2 in finals
            assert queue.submit(inputs[3]) == 3
            # Multiple complete chunks plus a tail may arrive in one fragment.
            queue.append(0, inputs[0][..., 2*width:], final=True)
            while queue.can_step: collect(queue.step())
            assert queue.pending == 0 and queue.buffered_bytes == 0
            assert queue.session.graph is captured
    assert sorted(finals) == [0, 1, 2, 3]
    for i, ref in enumerate(refs): assert torch.equal(torch.cat(results[i], -1), ref)


@torch.inference_mode()
def test_incremental_byte_capacity_input_ownership_validation_and_cancel():
    model = structural_model()
    x = torch.zeros(1, 3840, device='cuda')
    with optimized(model, kv_backend='triton'):
        with StreamingBatcher(model, 'encode', 2, 1, max_buffered_bytes=x.numel()*4) as queue:
            request = queue.submit(x, final=False)
            x.fill_(float('nan'))
            assert queue.buffered_bytes == 15360
            queue.step()  # Consumed prefix remains held by this fragment.
            assert queue.buffered_bytes == 15360
            with pytest.raises(BufferError): queue.append(request, torch.zeros(1, 1, device='cuda'), final=True)
            assert queue.buffered_bytes == 15360 and queue.pending == 1
            assert all(not c.final for c in queue.step())
            assert queue.buffered_bytes == 0 and not queue.can_step
            for bad in [torch.zeros(1, 1), torch.zeros(2, 1, device='cuda'),
                        torch.zeros(1, 1, device='cuda', dtype=torch.float64)]:
                with pytest.raises(ValueError): queue.append(request, bad)
            with pytest.raises(TypeError): queue.append(request, final=1)
            assert not queue.can_step
            queue.append(request, torch.zeros(1, 17, device='cuda'), final=True)
            with pytest.raises(RuntimeError): queue.append(request, final=True)
            queue.cancel(request)
            assert queue.buffered_bytes == 0
            with pytest.raises(KeyError): queue.append(request, final=True)
            empty = queue.submit(final=False)
            assert not queue.can_step and queue.step() == []
            queue.append(empty, final=True)
            assert queue.can_step and queue.step()[0].final
            assert not queue.pending and not queue.can_step
            waiting = queue.submit(final=False)
            other = torch.cuda.Stream()
            with torch.cuda.stream(other):
                with pytest.raises(RuntimeError, match='construction stream'): queue.append(waiting, final=True)
    assert queue.buffered_bytes == 0

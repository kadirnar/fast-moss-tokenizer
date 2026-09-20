"""Schedule complete or incrementally supplied requests in a fixed streaming batch."""
from collections import deque
from dataclasses import dataclass
from numbers import Integral

import torch
import triton
import triton.language as tl

from .streaming import StreamingSession


@triton.jit
def _gather(Meta, Out, WIDTH: tl.constexpr, CHANNELS: tl.constexpr,
            BATCH: tl.constexpr, ENCODE: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)
    channel = tl.program_id(1)
    t = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    pointer = tl.load(Meta + lane * 4)
    stride = tl.load(Meta + lane * 4 + 1)
    start = tl.load(Meta + lane * 4 + 2)
    length = tl.load(Meta + lane * 4 + 3)
    if ENCODE:
        source = pointer.to(tl.pointer_type(tl.float32))
        offset = lane * WIDTH + t
    else:
        source = pointer.to(tl.pointer_type(tl.int64))
        offset = (channel * BATCH + lane) * WIDTH + t
    value = tl.load(source + channel * stride + start + t,
                    (t < length) & (t < WIDTH), other=0)
    tl.store(Out + offset, value, t < WIDTH)


@dataclass(frozen=True)
class BatchChunk:
    """Owned output with request identity and zero-based output offset.

    data is (quantizers, frames) for encode or (1, samples) for decode.
    input_length counts samples/frames consumed by this chunk, not the whole
    request. Empty requests use lane=-1; all other chunks keep their lane.
    """
    request_id: int
    data: torch.Tensor
    offset: int
    final: bool
    input_length: int
    lane: int


@dataclass
class _Request:
    fragments: deque
    available: int
    final: bool
    head_offset: int = 0
    output_position: int = 0


@triton.jit(do_not_specialize=['SEGMENTS'])
def _gather_segmented(Meta, Out, WIDTH: tl.constexpr, BATCH: tl.constexpr,
                      SEGMENTS, ENCODE: tl.constexpr, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    channel = tl.program_id(1)
    segment = tl.load(Meta + SEGMENTS * 6 + task * 2)
    start = tl.load(Meta + SEGMENTS * 6 + task * 2 + 1)
    descriptor = Meta + segment * 6
    pointer = tl.load(descriptor)
    stride = tl.load(descriptor + 1)
    source_start = tl.load(descriptor + 2)
    destination = tl.load(descriptor + 3)
    length = tl.load(descriptor + 4)
    lane = tl.load(descriptor + 5)
    i = start + tl.arange(0, BLOCK)
    if ENCODE:
        source = pointer.to(tl.pointer_type(tl.float32))
        offset = lane * WIDTH + destination + i
    else:
        source = pointer.to(tl.pointer_type(tl.int64))
        offset = (channel * BATCH + lane) * WIDTH + destination + i
    value = tl.load(source + channel * stride + source_start + i,
                    (i < length) & (pointer != 0), 0)
    tl.store(Out + offset, value, i < length)


def _gather_chunks(segments, *, width, channels, encode, device):
    """Internal: lane segments partition valid input; synthesize all zero padding.

    Each source segment is (owned contiguous tensor, source offset, length).
    Segment tasks write disjoint output spans in one launch, without concatenating
    input fragments or looping over every fragment in every output tile.
    """
    batch = len(segments)
    shape = (batch, 1, width) if encode else (channels, batch, width)
    chunk = torch.empty(shape, device=device, dtype=torch.float32 if encode else torch.long)
    if all(len(parts) <= 1 for parts in segments):
        entries = [[parts[0][0].data_ptr(), parts[0][0].shape[-1], parts[0][1], parts[0][2]]
                   if parts else [0, 0, 0, 0] for parts in segments]
        meta = torch.tensor(entries, device=device, dtype=torch.long)
        _gather[(batch, channels, triton.cdiv(width, 256))](
            meta, chunk, width, channels, batch, encode, 256)
    else:
        entries, tasks = [], []
        for lane, parts in enumerate(segments):
            destination = 0
            for value, source_start, length in parts:
                index = len(entries)
                entries.append([value.data_ptr(), value.shape[-1], source_start, destination, length, lane])
                tasks.extend((index, start) for start in range(0, length, 256))
                destination += length
            if destination < width:
                index = len(entries)
                entries.append([0, 0, 0, destination, width - destination, lane])
                tasks.extend((index, start) for start in range(0, width - destination, 256))
        meta = torch.tensor([v for row in entries for v in row] + [v for row in tasks for v in row],
                            device=device, dtype=torch.long)
        _gather_segmented[(len(tasks), channels)](meta, chunk, width, batch, len(entries), encode, 256)
    return chunk


class StreamingBatcher:
    """Assign ready CUDA requests to stable lanes and refill finished slots.

    Use inside optimized(model, kv_backend='triton').
    submit(data) accepts complete unbatched (1, samples) FP32 audio or
    (quantizers, frames) int64 codes. For incremental input, submit(..., final=False)
    and append(request_id, fragment, final=...). append(id, final=True) finishes
    an input without adding data. Each nonempty submission owns a contiguous copy.

    A request waits for a full chunk unless its input is final. Waiting active
    requests keep their lane/history; unready queued requests do not block ready
    requests from entering free lanes. step() may return [] with pending requests;
    can_step tells callers when more input or completion is required. No model
    executes when every active request is waiting.

    max_pending bounds request count. Optional max_buffered_bytes bounds retained
    input tensor payloads, including consumed prefixes of partially held fragments;
    it excludes allocator overhead, model state, and caller-owned outputs.
    Calls use one host thread and the CUDA stream used on entry. A model may only
    belong to one active session. Fixed batch arithmetic is retained; cross-batch
    numerical equality must be tested separately. This is not a network server.
    """

    def __init__(self, model, direction, batch_size=8, chunk_frames=3,
                 use_graph=True, max_pending=128, max_buffered_bytes=None):
        for name, value in [('batch_size', batch_size), ('chunk_frames', chunk_frames),
                            ('max_pending', max_pending)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if max_buffered_bytes is not None and (not isinstance(max_buffered_bytes, Integral)
                or isinstance(max_buffered_bytes, bool) or max_buffered_bytes < 1):
            raise ValueError('max_buffered_bytes must be a positive integer or None')
        self.session = StreamingSession(model, direction, batch_size, chunk_frames, use_graph)
        self.model, self.direction = model, direction
        self.batch_size, self.max_pending = batch_size, max_pending
        self.max_buffered_bytes = max_buffered_bytes
        self._buffered_bytes = 0
        self.width = chunk_frames * (model.downsample_rate if direction == 'encode' else 1)
        self.channels = 1 if direction == 'encode' else model.quantizer.num_quantizers
        self.dtype = torch.float32 if direction == 'encode' else torch.long
        self._requests = {}
        self._queue = deque()
        self._lanes = [None] * batch_size
        self._dirty = [False] * batch_size
        self._next_id = 0
        self._entered = False
        self._used = False

    @torch.inference_mode()
    def __enter__(self):
        if self._used:
            raise RuntimeError('A batcher may only be entered once')
        device = next(self.model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('CUDA model required')
        self.session.__enter__()
        if not self.session._supports_pause:
            self.session.__exit__(None, None, None)
            raise ValueError("Batch scheduling requires optimized(model, kv_backend='triton')")
        self.device = device
        self.stream = torch.cuda.current_stream(device)
        self._entered = self._used = True
        return self

    def _check(self):
        if not self._entered:
            raise RuntimeError('Enter an active batcher first')
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError('Use this batcher on its construction stream')

    @property
    def pending(self):
        return len(self._requests)

    @property
    def buffered_bytes(self):
        return self._buffered_bytes

    def _ready(self, request):
        return request.available >= self.width or request.final

    @property
    def can_step(self):
        if any(r.final and not r.available for r in self._requests.values()):
            return True
        if any(i is not None and self._ready(self._requests[i]) for i in self._lanes):
            return True
        return (None in self._lanes and any(self._ready(self._requests[i]) for i in self._queue))

    def _own(self, data, final):
        if not isinstance(final, bool):
            raise TypeError('final must be a boolean')
        if data is None:
            return None
        if (not isinstance(data, torch.Tensor) or data.ndim != 2 or data.shape[0] != self.channels
                or data.dtype != self.dtype or data.device != self.device):
            raise ValueError(f'Expected CUDA {self.dtype} request of shape ({self.channels}, time)')
        size = data.numel() * data.element_size()
        if self.max_buffered_bytes is not None and self._buffered_bytes + size > self.max_buffered_bytes:
            raise BufferError('Input byte capacity reached; step or cancel before appending')
        return data.detach().clone(memory_format=torch.contiguous_format) if size else None

    @torch.inference_mode()
    def submit(self, data=None, *, final=True):
        self._check()
        if self.pending >= self.max_pending:
            raise BufferError('Request capacity reached; step or cancel before submitting')
        owned = self._own(data, final)
        request_id = self._next_id
        self._next_id += 1
        fragments = deque([owned]) if owned is not None else deque()
        self._requests[request_id] = _Request(fragments, owned.shape[-1] if owned is not None else 0, final)
        self._buffered_bytes += owned.numel() * owned.element_size() if owned is not None else 0
        self._queue.append(request_id)
        return request_id

    @torch.inference_mode()
    def append(self, request_id, data=None, *, final=False):
        self._check()
        request = self._requests[request_id]
        if request.final:
            raise RuntimeError('Request input is already final')
        owned = self._own(data, final)
        if owned is not None:
            request.fragments.append(owned)
            request.available += owned.shape[-1]
            self._buffered_bytes += owned.numel() * owned.element_size()
        request.final = final

    def cancel(self, request_id):
        self._check()
        request = self._requests.pop(request_id)
        self._buffered_bytes -= sum(x.numel() * x.element_size() for x in request.fragments)
        for lane, current in enumerate(self._lanes):
            if current == request_id:
                self._lanes[lane] = None
                self._dirty[lane] = True
        self._queue = deque(i for i in self._queue if i != request_id)

    def _consume(self, request, length):
        request.available -= length
        while length:
            fragment = request.fragments[0]
            remaining = fragment.shape[-1] - request.head_offset
            if length < remaining:
                request.head_offset += length
                break
            length -= remaining
            request.fragments.popleft()
            request.head_offset = 0
            self._buffered_bytes -= fragment.numel() * fragment.element_size()

    @torch.inference_mode()
    def step(self):
        self._check()
        outputs = []
        # Also handles final notification after the last full chunk was already
        # emitted. Free state is marked dirty and resets before any later reuse.
        for request_id, request in list(self._requests.items()):
            if request.final and not request.available:
                lane = self._lanes.index(request_id) if request_id in self._lanes else -1
                shape = (self.model.quantizer.num_quantizers, 0) if self.direction == 'encode' else (1, 0)
                dtype = torch.long if self.direction == 'encode' else torch.float32
                outputs.append(BatchChunk(request_id, torch.empty(shape, device=self.device, dtype=dtype),
                                          request.output_position, True, 0, lane))
                self.cancel(request_id)
        free = deque(i for i, request_id in enumerate(self._lanes) if request_id is None)
        waiting = deque(); reset = [False] * self.batch_size
        for request_id in self._queue:
            if free and self._ready(self._requests[request_id]):
                lane = free.popleft()
                self._lanes[lane] = request_id
                reset[lane] = self._dirty[lane]
                self._dirty[lane] = False
            else:
                waiting.append(request_id)
        self._queue = waiting
        lengths, ends, segments = [], [], []
        for request_id in self._lanes:
            request = self._requests[request_id] if request_id is not None else None
            n = min(self.width, request.available) if request is not None and self._ready(request) else 0
            lengths.append(n)
            ends.append(bool(n and request.final and n == request.available))
            parts = []; remaining = n
            if remaining:
                for index, fragment in enumerate(request.fragments):
                    start = request.head_offset if index == 0 else 0
                    take = min(fragment.shape[-1] - start, remaining)
                    parts.append((fragment, start, take))
                    remaining -= take
                    if not remaining: break
            segments.append(parts)
        if not any(lengths):
            return outputs
        if any(reset):
            self.session.reset(torch.tensor(reset, device=self.device, dtype=torch.bool))
        chunk = _gather_chunks(segments, width=self.width, channels=self.channels,
                               encode=self.direction == 'encode', device=self.device)
        data, _ = self.session.push(chunk, valid_lengths=lengths, final_lanes=ends)
        for lane, request_id in enumerate(self._lanes):
            n = lengths[lane]
            if not n:
                continue
            request = self._requests[request_id]
            valid = ((n + self.model.downsample_rate - 1) // self.model.downsample_rate
                     if self.direction == 'encode' else n * self.model.downsample_rate)
            view = data[:, lane, :valid] if self.direction == 'encode' else data[lane, :, :valid]
            outputs.append(BatchChunk(request_id, view.clone(), request.output_position,
                                      ends[lane], n, lane))
            self._consume(request, n)
            request.output_position += valid
            if ends[lane]:
                self.cancel(request_id)
        return outputs

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.session.__exit__(exc_type, exc_value, traceback)
        finally:
            self._entered = False
            self._requests.clear()
            self._queue.clear()
            self._lanes = [None] * self.batch_size
            self._buffered_bytes = 0

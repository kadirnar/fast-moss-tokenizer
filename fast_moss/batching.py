"""FIFO scheduling of finite requests through a reusable streaming batch."""
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
    data: torch.Tensor
    position: int = 0
    output_position: int = 0


class StreamingBatcher:
    """Assign finite CUDA requests to stable lanes and refill finished slots.

    Use inside optimized(model, kv_backend='triton'). submit() takes unbatched
    (1, samples) FP32 audio or (quantizers, frames) int64 codes and owns a copy.
    step() returns the next available chunk from each active request, with final
    tails trimmed. New requests may be submitted between steps. Queue capacity
    counts active and waiting requests. cancel() discards the remaining input.

    Calls must use one host thread and the CUDA stream used on entry. This is a
    finite-request scheduler, not a network server or an incremental input queue.
    A model may only belong to one active session. Arithmetic uses the configured
    fixed batch shape; cross-batch numerical equality must be tested separately.
    """

    def __init__(self, model, direction, batch_size=8, chunk_frames=3,
                 use_graph=True, max_pending=128):
        for name, value in [('batch_size', batch_size), ('chunk_frames', chunk_frames),
                            ('max_pending', max_pending)]:
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.session = StreamingSession(model, direction, batch_size, chunk_frames, use_graph)
        self.model, self.direction = model, direction
        self.batch_size, self.max_pending = batch_size, max_pending
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

    @torch.inference_mode()
    def submit(self, data):
        self._check()
        if (data.ndim != 2 or data.shape[0] != self.channels
                or data.dtype != self.dtype or data.device != self.device):
            raise ValueError(f'Expected CUDA {self.dtype} request of shape ({self.channels}, time)')
        if self.pending >= self.max_pending:
            raise BufferError('Request capacity reached; step or cancel before submitting')
        owned = data.detach().clone(memory_format=torch.contiguous_format)
        request_id = self._next_id
        self._next_id += 1
        self._requests[request_id] = _Request(owned)
        self._queue.append(request_id)
        return request_id

    def cancel(self, request_id):
        self._check()
        if request_id not in self._requests:
            raise KeyError(request_id)
        del self._requests[request_id]
        for lane, current in enumerate(self._lanes):
            if current == request_id:
                self._lanes[lane] = None
                self._dirty[lane] = True
        # Remove queued IDs now so cancellation cannot grow an unbounded backlog.
        self._queue = deque(i for i in self._queue if i != request_id)

    @torch.inference_mode()
    def step(self):
        self._check()
        outputs = []
        # Empty requests complete without capturing a graph or occupying a lane.
        waiting = deque()
        for request_id in self._queue:
            request = self._requests[request_id]
            if request.data.shape[-1] == 0:
                shape = (self.model.quantizer.num_quantizers, 0) if self.direction == 'encode' else (1, 0)
                dtype = torch.long if self.direction == 'encode' else torch.float32
                outputs.append(BatchChunk(request_id, torch.empty(shape, device=self.device, dtype=dtype),
                                          0, True, 0, -1))
                del self._requests[request_id]
            else:
                waiting.append(request_id)
        self._queue = waiting
        reset = [False] * self.batch_size
        for lane in range(self.batch_size):
            if self._lanes[lane] is None and self._queue:
                self._lanes[lane] = self._queue.popleft()
                reset[lane] = self._dirty[lane]
                self._dirty[lane] = False
        if not any(i is not None for i in self._lanes):
            return outputs
        if any(reset):
            self.session.reset(torch.tensor(reset, device=self.device, dtype=torch.bool))
        metadata, lengths, ends = [], [], []
        for request_id in self._lanes:
            if request_id is None:
                metadata.append([0, 0, 0, 0]); lengths.append(0); ends.append(False)
            else:
                request = self._requests[request_id]
                n = min(self.width, request.data.shape[-1] - request.position)
                metadata.append([request.data.data_ptr(), request.data.shape[-1], request.position, n])
                lengths.append(n)
                ends.append(request.position + n == request.data.shape[-1])
        meta = torch.tensor(metadata, device=self.device, dtype=torch.long)
        shape = ((self.batch_size, 1, self.width) if self.direction == 'encode'
                 else (self.channels, self.batch_size, self.width))
        chunk = torch.empty(shape, device=self.device, dtype=self.dtype)
        _gather[(self.batch_size, self.channels, triton.cdiv(self.width, 256))](
            meta, chunk, self.width, self.channels, self.batch_size, self.direction == 'encode', 256)
        data, _ = self.session.push(chunk, valid_lengths=lengths, final_lanes=ends)
        for lane, request_id in enumerate(self._lanes):
            if request_id is None:
                continue
            request = self._requests[request_id]
            n = lengths[lane]
            valid = ((n + self.model.downsample_rate - 1) // self.model.downsample_rate
                     if self.direction == 'encode' else n * self.model.downsample_rate)
            view = data[:, lane, :valid] if self.direction == 'encode' else data[lane, :, :valid]
            outputs.append(BatchChunk(request_id, view.clone(), request.output_position,
                                      ends[lane], n, lane))
            request.position += n
            request.output_position += valid
            if ends[lane]:
                del self._requests[request_id]
                self._lanes[lane] = None
                self._dirty[lane] = True
        return outputs

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self.session.__exit__(exc_type, exc_value, traceback)
        finally:
            self._entered = False
            self._requests.clear()
            self._queue.clear()
            self._lanes = [None] * self.batch_size

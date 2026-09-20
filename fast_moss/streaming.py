"""Incremental batched streaming with optional fixed-shape CUDA graphs."""
from contextlib import ExitStack
from numbers import Integral

import torch
import torch.nn.functional as F

from .graphs import GraphedCallable


class StreamingSession:
    """Context-managed encoder or decoder session, yielding each chunk immediately.

    Active lanes advance one fixed chunk; paused lanes keep history and return
    length zero. Host sequences valid_lengths/final_lanes permit different final
    tails and independent completion. Completed lanes stay paused until reset.
    Use separate model instances for overlapping sessions. This class owns
    upstream cache state.
    """

    def __init__(self, model, direction, batch_size=1, chunk_frames=1, use_graph=True, fast_reset=True):
        if direction not in {"encode", "decode"}:
            raise ValueError("direction must be encode or decode")
        if batch_size < 1 or chunk_frames < 1:
            raise ValueError("batch_size and chunk_frames must be positive")
        if chunk_frames * model.downsample_rate > model.sampling_rate * model.causal_transformer_context_duration:
            raise ValueError("Chunk cannot exceed the attention context")
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Streaming requires a frozen eval model")
        if any(getattr(m, "weights_per_step", 0) for m in model.modules()):
            raise ValueError("Graph streaming does not support per-step weights")
        self.model, self.direction = model, direction
        self.batch_size, self.chunk_frames = batch_size, chunk_frames
        self.use_graph = use_graph
        self.fast_reset = fast_reset
        self._reset_plan = None
        self.active = False
        self.finished = False
        self.graph = None
        self._has_custom_mask = False

    @torch.inference_mode()
    def __enter__(self):
        if self.active or getattr(self.model, "_fast_streaming_owner", None) is not None:
            raise RuntimeError("Model already has an active streaming session")
        if any(getattr(m, "_streaming_state", None) is not None for m in self.model.modules()):
            raise RuntimeError("Upstream streaming state is already active")
        self.model._fast_streaming_owner = self
        self.stack = ExitStack()
        self.modules = getattr(self.model, "encoder" if self.direction == "encode" else "decoder")
        try:
            device = next(self.model.parameters()).device
            self._active = torch.ones(self.batch_size, device=device, dtype=torch.bool)
            self._closed = torch.zeros_like(self._active)
            self._uses_lane_controls = False
            self._supports_pause = True
            tokens = self.chunk_frames * (self.model.downsample_rate if self.direction == "encode" else 1)
            for module in self.modules:
                if hasattr(module, "streaming"):
                    self.stack.enter_context(module.streaming(self.batch_size))
                    # All keys for a chunk are inserted before any of its queries.
                    # Keep context-1 previous keys plus the complete new chunk.
                    # The unchanged attention mask still limits each query to context.
                    for child in module.modules():
                        state = getattr(child, "_streaming_state", None)
                        if state is not None:
                            # One stable mask address for all layers; graph replay
                            # reads it without hundreds of per-layer mask copies.
                            state.exec_mask = self._active
                            # This session applies resets/masks to every layer,
                            # so all layer offsets within a stage stay aligned.
                            state._fast_synchronized = True
                        cache = getattr(state, "kv_cache", None)
                        if cache is not None:
                            self._supports_pause &= hasattr(child, "_fast_original_complete")
                        if cache is not None and tokens > 1:
                            cache.capacity += tokens - 1
                            shape = list(cache.cache.shape)
                            shape[-2] = cache.capacity
                            cache.cache = torch.zeros(shape, device=cache.cache.device, dtype=cache.cache.dtype)
                if hasattr(module, "patch_size"):
                    if self.direction == "encode":
                        tokens //= module.patch_size
                    else:
                        tokens *= module.patch_size
            if self.fast_reset and self._active.is_cuda:
                from .stream_reset import ResetPlan
                states = [child._streaming_state for root in self.modules for child in root.modules()
                          if getattr(child, '_streaming_state', None) is not None]
                try:
                    self._reset_plan = ResetPlan(states, self._active, self._closed)
                except ValueError:
                    self._reset_plan = None
            self.active = True
            self.finished = False
            self._has_custom_mask = False
            return self
        except BaseException:
            self.stack.close()
            self._reset_plan = None
            del self.model._fast_streaming_owner
            raise

    def _run(self, chunk, lengths):
        if self.direction == "encode":
            result = self.model._encode_frame(chunk, lengths)
            return result.audio_codes, result.audio_codes_lengths
        result = self.model._decode_frame(chunk, lengths)
        return result.audio, result.audio_lengths

    @torch.inference_mode()
    def _check_mask(self, mask):
        if (mask.shape != (self.batch_size,) or mask.dtype != torch.bool
                or mask.device != self._active.device):
            raise ValueError("Expected a CUDA boolean mask of shape (batch_size,)")

    @torch.inference_mode()
    def reset(self, mask=None):
        if not self.active:
            raise RuntimeError("Session is not active")
        self._check_stream()
        device = next(self.model.parameters()).device
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=device)
            self._has_custom_mask = False
        else:
            self._check_mask(mask)
            self._has_custom_mask = True
        if self._reset_plan is not None:
            self._reset_plan.reset(mask)
        else:
            for root in self.modules:
                for module in root.modules():
                    state = getattr(module, "_streaming_state", None)
                    if state is not None:
                        state.reset(mask)
            self._closed.masked_fill_(mask, False)
        if not self._has_custom_mask:
            self._uses_lane_controls = False
        self.finished = False

    @torch.inference_mode()
    def push(self, chunk, *, final=False, active_mask=None, valid_lengths=None, final_lanes=None):
        """Return owned outputs and per-lane valid lengths on the GPU.

        valid_lengths is a host sequence of sample counts (encode) or code-frame
        counts (decode). Zero pauses a lane. A short positive length requires its
        final_lanes flag or final=True. Padding beyond each valid length is ignored.
        final_lanes is a host bool sequence: flags close lanes after this push,
        even when empty or inactive. Completed lanes ignore input until reset.
        final=True still finishes the entire session. Trim each output lane by
        its returned length; save original sample counts for waveform trimming.
        """
        if not self.active or self.finished:
            raise RuntimeError("Enter an active session and do not push after final=True")
        self._check_stream()
        if not isinstance(final, bool):
            raise TypeError("final must be a boolean")
        expected = self.chunk_frames * (self.model.downsample_rate if self.direction == "encode" else 1)
        shape = ((self.batch_size, 1) if self.direction == "encode"
                 else (self.model.quantizer.num_quantizers, self.batch_size))
        dtype = torch.float32 if self.direction == "encode" else torch.long
        if chunk.ndim != 3 or chunk.shape[:2] != shape or chunk.dtype != dtype:
            raise ValueError(f"Expected shape {shape} + (time,) and dtype {dtype}")
        if chunk.device != next(self.model.parameters()).device:
            raise ValueError("Chunk and model must share a device")
        if active_mask is not None:
            self._check_mask(active_mask)
            if not self._supports_pause:
                raise ValueError("Paused lanes require optimized(model, kv_backend='triton')")
        length = chunk.shape[-1]
        if valid_lengths is not None or final_lanes is not None or self._uses_lane_controls:
            return self._push_lanes(chunk, expected, final, active_mask, valid_lengths, final_lanes)
        if length <= 0 or length > expected or (length != expected and not final):
            raise ValueError("Full chunks required except for a shorter final chunk")
        # Upstream floors lengths at each patching stage. Mark a partial final
        # codec frame as valid after zero padding so its audio is not discarded.
        effective_length = length
        if self.direction == "encode":
            rate = self.model.downsample_rate
            effective_length = ((length + rate - 1) // rate) * rate
        lengths = torch.full((self.batch_size,), effective_length, device=chunk.device, dtype=torch.long)
        if active_mask is not None:
            lengths = lengths * active_mask
        if length < expected:
            chunk = F.pad(chunk, (0, expected - length))
        if self.use_graph:
            if self.graph is None:
                self.graph = GraphedCallable(self._run, chunk, lengths)
                self.reset()  # Discard warmup/capture history, keep graph tensor addresses.
            self._set_active(active_mask)
            outputs = self.graph(chunk, lengths)
        else:
            self._set_active(active_mask)
            outputs = self._run(chunk, lengths)
        self.finished = final
        # Trim using host-known shape; no .item() or device synchronization.
        valid = ((length + self.model.downsample_rate - 1) // self.model.downsample_rate
                 if self.direction == "encode" else length * self.model.downsample_rate)
        return outputs[0][..., :valid], outputs[1]

    def _push_lanes(self, chunk, expected, final, active_mask, valid_lengths, final_lanes):
        """Host-known metadata controls lengths; persistent completion stays on GPU."""
        if not self._supports_pause:
            raise ValueError("Per-lane completion requires optimized(model, kv_backend='triton')")
        if isinstance(valid_lengths, torch.Tensor) or isinstance(final_lanes, torch.Tensor):
            raise TypeError("valid_lengths and final_lanes must be host sequences, not tensors")
        n = tuple(valid_lengths) if valid_lengths is not None else (chunk.shape[-1],) * self.batch_size
        ends = tuple(final_lanes) if final_lanes is not None else (False,) * self.batch_size
        if len(n) != self.batch_size or len(ends) != self.batch_size:
            raise ValueError("One length and completion flag per batch lane required")
        if any(not isinstance(v, Integral) or isinstance(v, bool) for v in n):
            raise TypeError("valid_lengths must contain integers")
        if any(not isinstance(v, bool) for v in ends):
            raise TypeError("final_lanes must contain booleans")
        ends = tuple(end or final for end in ends)
        if chunk.shape[-1] > expected or any(v < 0 or v > chunk.shape[-1] for v in n):
            raise ValueError("Lane lengths must be within the provided chunk and configured width")
        if any(0 < v < expected and not end for v, end in zip(n, ends)):
            raise ValueError("A short nonempty lane must finish in this push")
        if not any(n):
            self._active.zero_()
            self._finish_lanes(ends, final)
            shape = ((self.model.quantizer.num_quantizers, self.batch_size, 0)
                     if self.direction == "encode" else (self.batch_size, 1, 0))
            dtype = torch.long if self.direction == "encode" else torch.float32
            return (torch.empty(shape, device=chunk.device, dtype=dtype),
                    torch.zeros(self.batch_size, device=chunk.device, dtype=torch.long))
        from .stream_inputs import prepare
        raw = torch.tensor(n, device=chunk.device, dtype=torch.long)
        prepared, lengths = prepare(chunk, raw, active_mask, self._closed, self._active,
                                    encode=self.direction == "encode", width=expected,
                                    rate=self.model.downsample_rate)
        if self.use_graph:
            if self.graph is None:
                initial_active = self._active.clone()
                initial_closed = self._closed.clone()
                self.graph = GraphedCallable(self._run, prepared, lengths)
                self.reset()
                self._active.copy_(initial_active)
                self._closed.copy_(initial_closed)
            outputs = self.graph(prepared, lengths)
        else:
            outputs = self._run(prepared, lengths)
        self._finish_lanes(ends, final)
        valid = ((max(n) + self.model.downsample_rate - 1) // self.model.downsample_rate
                 if self.direction == "encode" else max(n) * self.model.downsample_rate)
        return outputs[0][..., :valid], outputs[1]

    def _finish_lanes(self, ends, final):
        if any(ends):
            self._closed.logical_or_(torch.tensor(ends, device=self._closed.device, dtype=torch.bool))
        self._uses_lane_controls = True
        self._has_custom_mask = True
        self.finished = final

    def _check_stream(self):
        if self.graph is not None and torch.cuda.current_stream(self.graph.device) != self.graph.stream:
            raise RuntimeError("Use this session on its graph construction stream")

    def _set_active(self, active_mask):
        if active_mask is not None:
            self._active.copy_(active_mask)
            self._has_custom_mask = True
        elif self._has_custom_mask:
            self._active.fill_(True)
            self._has_custom_mask = False

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stack.close()
        finally:
            self.graph = None
            self._reset_plan = None
            self.active = False
            del self.model._fast_streaming_owner

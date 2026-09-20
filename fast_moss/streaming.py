"""CUDA graph replay with the native v2 streaming attention history."""

from contextlib import ExitStack
from threading import get_ident
from types import MethodType

import torch

from .graphs import GraphedCallable
from .v2_codec import _encode_complete
from .v2_runtime import validate


def _stable_cache_update(original):
    def update(module, state, keys, values, positions, *current):
        # Keep upstream selection and arithmetic, but retain the addresses that
        # the next graph replay reads. Replacing state tensors freezes history.
        original(state, keys, values, positions, *current)
        keys.copy_(state.cached_keys)
        values.copy_(state.cached_values)
        positions.copy_(state.cached_positions)
        state.cached_keys = keys
        state.cached_values = values
        state.cached_positions = positions

    return update


class StreamingCodec:
    """Reusable full-file codec for B=1, FP32, 48 kHz stereo CUDA inputs.

    Enter inside ``optimized(model)``. Each call starts a fresh file, preserves
    native history across 80 ms chunks, and returns codes, hidden states, audio.
    Setup is paid once; model weights and settings must remain fixed until exit.
    Calls and context management must use the construction thread and stream.
    """

    def __init__(self, model):
        validate(model)
        self.model = model
        self.device = next(model.parameters()).device
        self.thread = get_ident()
        self.stream = torch.cuda.current_stream(self.device)
        self.stack = None
        self.graph = None
        self.states = []
        self.used = False

    def _check_thread(self):
        if (
            get_ident() != self.thread
            or torch.cuda.current_stream(self.device) != self.stream
        ):
            raise RuntimeError(
                "Use StreamingCodec on its construction thread and CUDA stream"
            )

    @torch.inference_mode()
    def __enter__(self):
        self._check_thread()
        if self.used:
            raise RuntimeError("Create a new StreamingCodec instead of re-entering")
        if any(getattr(m, "is_streaming", False) for m in self.model.modules()):
            raise RuntimeError("The model already has an active streaming session")
        self.used = True
        self.stack = ExitStack()
        try:
            for module in self.model.modules():
                if not hasattr(module, "_update_streaming_cache"):
                    continue
                if module.context is None:
                    raise ValueError("Streaming graphs require bounded attention history")
                name = "_update_streaming_cache"
                original = getattr(module, name)
                if name in module.__dict__:
                    self.stack.callback(setattr, module, name, module.__dict__[name])
                else:
                    self.stack.callback(delattr, module, name)
                replacement = MethodType(_stable_cache_update(original), module)
                setattr(module, name, replacement)
            # Also clean up if native state allocation fails before yielding.
            self.stack.callback(self.model._stop_streaming)
            self.stack.enter_context(self.model.streaming(batch_size=1))
            self.states = [
                m._streaming_state
                for m in self.model.modules()
                if getattr(m, "_streaming_state", None) is not None
            ]
            example = torch.zeros(1, 2, 3840, device=self.device)

            def step(audio):
                encoded = _encode_complete(self.model, audio)
                decoded = self.model._decode_frame(encoded.audio_codes)
                return encoded.audio_codes, encoded.encoder_hidden_states, decoded.audio

            self.graph = GraphedCallable(step, example)
            self._reset()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _reset(self):
        # Stereo is interleaved into the model's channel dimension; each native
        # state owns its actual batch size. Reset contents, never replace storage.
        for state in self.states:
            state.reset(torch.ones_like(state.exec_mask))

    @torch.inference_mode()
    def __call__(self, audio):
        self._check_thread()
        if self.graph is None:
            raise RuntimeError("Use StreamingCodec inside its active context")
        if (
            audio.ndim != 3
            or audio.shape[:2] != (1, 2)
            or audio.shape[-1] < 1
            or audio.device != self.device
            or audio.dtype != torch.float32
        ):
            raise ValueError(
                "Expected nonempty FP32 CUDA audio with shape (1, 2, samples)"
            )
        self._reset()
        length = audio.shape[-1]
        padding = (-length) % 3840
        if padding:
            audio = torch.nn.functional.pad(audio, (0, padding))
        chunks = [self.graph(chunk) for chunk in audio.split(3840, dim=-1)]
        codes, hidden, reconstructed = (
            torch.cat(parts, dim=-1) for parts in zip(*chunks)
        )
        return codes, hidden, reconstructed[..., :length]

    def __exit__(self, *exc):
        self._check_thread()
        torch.cuda.synchronize(self.device)
        self.graph = None
        self.states = []
        if self.stack is not None:
            self.stack.close()
            self.stack = None

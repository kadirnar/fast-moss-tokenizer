"""Reversible optimizations for the pinned 48 kHz stereo v2 checkpoint."""

from contextlib import contextmanager
from threading import get_ident
from types import MethodType
import math

import torch
import torch.nn.functional as F
from torch.nn.modules import module as module_hooks

from . import storage_epoch
from .v2_loading import REVISION

_LINEAR_FORWARD = torch.nn.Linear.forward


def validate(model):
    if (
        type(model).__name__ != "MossAudioTokenizerModel"
        or getattr(model.config, "_commit_hash", None) != REVISION
        or model.sampling_rate != 48000
        or model.number_channels != 2
        or model.downsample_rate != 3840
        or not model.enable_channel_interleave
    ):
        raise ValueError("Expected the pinned 48 kHz stereo MOSS v2 checkpoint")
    device = next(model.parameters()).device
    if device.type != "cuda" or torch.__version__ != "2.8.0+cu128":
        raise ValueError(
            "V2 optimizations require CUDA and the validated PyTorch 2.8.0+cu128"
        )
    props = torch.cuda.get_device_properties(device)
    if (
        props.name != "NVIDIA GeForce RTX 5070 Ti"
        or (props.major, props.minor) != (12, 0)
        or props.multi_processor_count != 70
    ):
        raise ValueError("V2 kernels require the validated RTX 5070 Ti")
    if model.attention_implementation != "sdpa":
        raise ValueError("The validated v2 attention backend is SDPA")


class V2Runtime:
    """Single-host-thread owner; original FP32 parameters remain in place."""

    def __init__(self, model, *, cache_linear=True, quantizer=True, pointwise=True):
        validate(model)
        self.model = model
        self.device = next(model.parameters()).device
        self.thread = get_ident()
        self.cache_linear = cache_linear
        self.quantizer = quantizer
        self.pointwise = pointwise
        self.active = self.used = False
        self.saved = []
        self.warmed = set()
        self.cached_bytes = self.linear_calls = self.prepare_calls = (
            self.select_calls
        ) = 0
        self.linear_modules = self.conv_modules = self.quantizer_modules = 0
        self.rope_modules = self.residual_modules = self.rope_calls = (
            self.residual_calls
        ) = 0
        self.frequencies = {}

    def _check(self):
        if not self.active or get_ident() != self.thread:
            raise RuntimeError("Use v2 optimizations on their active owner thread")

    def _replace(self, module, name, value):
        self.saved.append(
            (module, name, name in module.__dict__, module.__dict__.get(name))
        )
        setattr(module, name, value)

    def _linear(self, original, weight, bias, cached_weight, cached_bias):
        versions = (weight._version, bias._version if bias is not None else None)

        def forward(module, x):
            self._check()
            if (
                module.weight is not weight
                or module.bias is not bias
                or weight._version != versions[0]
                or (bias is not None and bias._version != versions[1])
                or x.device != self.device
                or x.dtype not in (torch.float32, torch.bfloat16)
                or not torch.is_autocast_enabled("cuda")
                or torch.get_autocast_dtype("cuda") != torch.bfloat16
                or (
                    torch.is_grad_enabled()
                    and (
                        x.requires_grad
                        or weight.requires_grad
                        or (bias is not None and bias.requires_grad)
                    )
                )
            ):
                return original(x)
            self.linear_calls += 1
            # Let the same native autocast/F.linear path cast the activation.
            # The immutable weight cast has already been computed exactly once.
            return F.linear(x, cached_weight, cached_bias)

        return forward

    def _conv(self, original, weight, parameters):
        versions = tuple(p._version for p in parameters)

        def forward(module, x):
            self._check()
            current = (
                tuple(module.parametrizations.weight.parameters())
                if torch.nn.utils.parametrize.is_parametrized(module, "weight")
                else ()
            )
            if (
                len(current) != len(parameters)
                or any(a is not b for a, b in zip(current, parameters))
                or any(
                    p._version != version for p, version in zip(parameters, versions)
                )
                or module_hooks._global_forward_hooks
                or module_hooks._global_forward_pre_hooks
                or any(
                    m._forward_hooks or m._forward_pre_hooks
                    for m in module.parametrizations.modules()
                )
                or (
                    torch.is_grad_enabled()
                    and (x.requires_grad or any(p.requires_grad for p in parameters))
                )
            ):
                return original(x)
            return module._conv_forward(x, weight, module.bias)

        return forward

    def _decode(self, original, weight, book, norm):
        version = weight._version

        def decode(module, x):
            self._check()
            if (
                module.codebook.weight is not weight
                or weight._version != version
                or x.ndim != 3
                or x.shape[1] != 8
                or x.device != self.device
                or x.dtype != torch.float32
                or x.shape[0] < 1
                or x.shape[2] < 1
                or torch.is_autocast_enabled("cuda")
                or (
                    torch.is_grad_enabled()
                    and (x.requires_grad or weight.requires_grad)
                )
            ):
                return original(x)
            if x.is_contiguous() and x.numel() <= 2147483647:
                from .quantizer_prepare import prepare

                stream = torch.cuda.current_stream(self.device).cuda_stream
                key = (id(module), tuple(x.shape), stream)
                if torch.cuda.is_current_stream_capturing() and key not in self.warmed:
                    raise RuntimeError(
                        "Warm v2 quantizer preparation on this stream before capture"
                    )
                _, row_norm, twice, _ = prepare(x)
                self.warmed.add(key)
                self.prepare_calls += 1
            else:
                enc = F.normalize(x.transpose(1, 2).reshape(-1, 8).float())
                row_norm = enc.pow(2).sum(1, keepdim=True)
                twice = 2 * enc
            from .quantizer import select

            dots = twice @ book.T
            self.select_calls += 1
            return select(dots, row_norm, norm, weight, x)

        return decode

    def _rope(self, original, period):
        def forward(module, q, k, offset, time_before_heads=False):
            self._check()
            if (
                time_before_heads
                or q.ndim != 4
                or q.shape != k.shape
                or q.shape[-1] % 2
                or q.shape[-1] < 2
                or q.dtype not in (torch.bfloat16, torch.float32)
                or k.dtype != q.dtype
                or q.device != self.device
                or k.device != self.device
                or offset.device != self.device
                or offset.numel() != q.shape[0]
                or module.max_period != period
                or (
                    torch.is_grad_enabled()
                    and any(t.requires_grad for t in (q, k, offset))
                )
            ):
                return original(q, k, offset, time_before_heads)
            from .v2_pointwise import rotate

            b, h, t, d = q.shape
            key = (d, period)
            if key not in self.frequencies:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm v2 rotary frequencies before capture")
                ds = torch.arange(d // 2, device=self.device, dtype=torch.float32)
                self.frequencies[key] = torch.exp(ds * (-math.log(period) * 2 / d))
                self.cached_bytes += d // 2 * 4
            # Native PyTorch trigonometry, with the exact original FP32 phase.
            ts = offset.float().view(-1, 1) + torch.arange(
                t, device=self.device, dtype=torch.float32
            )
            phase = self.frequencies[key] * ts.view(b, t, 1)
            cos, sin = torch.cos(phase), torch.sin(phase)
            self.rope_calls += 1
            return rotate(q, k, cos, sin)

        return forward

    def _layer(self, original, scales):
        forwards = tuple(type(s).forward for s in scales)

        def forward(module, x, **kwargs):
            self._check()
            if (
                torch.is_grad_enabled()
                or x.device != self.device
                or x.dtype not in (torch.bfloat16, torch.float32)
                or module.layer_scale_1 is not scales[0]
                or module.layer_scale_2 is not scales[1]
                or module_hooks._global_forward_hooks
                or module_hooks._global_forward_pre_hooks
                or any(
                    s._forward_hooks
                    or s._forward_pre_hooks
                    or not s.channel_last
                    or getattr(s.forward, "__func__", None) is not fn
                    or s.scale.dtype != torch.float32
                    or s.scale.device != self.device
                    for s, fn in zip(scales, forwards)
                )
            ):
                return original(x, **kwargs)
            from .v2_pointwise import scale_add

            for norm, block, scale in [
                (module.norm1, module.self_attn, scales[0]),
                (module.norm2, module.ffn, scales[1]),
            ]:
                residual = x
                normed = norm(x)
                update = (
                    block(normed, **kwargs)
                    if block is module.self_attn
                    else block(normed)
                )
                if (
                    normed.dtype == torch.float32
                    and update.dtype in (torch.bfloat16, torch.float32)
                    and residual.shape == update.shape
                    and scale.scale.shape == (x.shape[-1],)
                    and all(t.is_contiguous() for t in (residual, update, scale.scale))
                ):
                    x = scale_add(residual, update, scale.scale)
                    self.residual_calls += 1
                else:
                    x = residual.to(normed) + scale(update)
            return x

        return forward

    def __enter__(self):
        if (
            self.used
            or getattr(self.model, "_fast_v2_runtime", None) is not None
            or getattr(self.model, "_fast_optimization_active", False)
        ):
            raise RuntimeError(
                "V2 optimization context is already active or has been used"
            )
        if get_ident() != self.thread:
            raise RuntimeError("Enter v2 optimization on its construction host thread")
        if self.model.training or any(
            p.requires_grad or p.dtype != torch.float32 or p.device != self.device
            for p in self.model.parameters()
        ):
            raise ValueError(
                "V2 optimizations require a frozen FP32 checkpoint on one CUDA device"
            )
        if any(getattr(m, "is_streaming", False) for m in self.model.modules()):
            raise RuntimeError("Enter v2 optimization before starting streaming")
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Enter v2 optimization before CUDA graph capture")
            torch.cuda.synchronize(self.device)
            storage_epoch.advance(self.device, context=True)
        self.active = self.used = True
        self.model._fast_v2_runtime = self
        try:
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                for module in self.model.modules():
                    if (
                        self.pointwise
                        and type(module).__name__ == "MossAudioTokenizerRotaryEmbedding"
                        and getattr(module.forward, "__func__", None)
                        is type(module).forward
                    ):
                        self._replace(
                            module,
                            "forward",
                            MethodType(
                                self._rope(module.forward, module.max_period), module
                            ),
                        )
                        self.rope_modules += 1
                    if (
                        self.pointwise
                        and type(module).__name__
                        == "MossAudioTokenizerTransformerLayer"
                        and getattr(module.forward, "__func__", None)
                        is type(module).forward
                        and type(module.norm1) is torch.nn.LayerNorm
                        and type(module.norm2) is torch.nn.LayerNorm
                        and all(
                            type(s).__name__ == "MossAudioTokenizerLayerScale"
                            for s in (module.layer_scale_1, module.layer_scale_2)
                        )
                    ):
                        self._replace(
                            module,
                            "forward",
                            MethodType(
                                self._layer(
                                    module.forward,
                                    (module.layer_scale_1, module.layer_scale_2),
                                ),
                                module,
                            ),
                        )
                        self.residual_modules += 1
                    if (
                        self.cache_linear
                        and type(module) is torch.nn.Linear
                        and torch.nn.Linear.forward is _LINEAR_FORWARD
                        and getattr(module.forward, "__func__", None) is _LINEAR_FORWARD
                    ):
                        weight, bias = module.weight, module.bias
                        cached_weight = weight.to(torch.bfloat16)
                        cached_bias = (
                            bias.to(torch.bfloat16) if bias is not None else None
                        )
                        self.cached_bytes += cached_weight.numel() * 2 + (
                            cached_bias.numel() * 2 if cached_bias is not None else 0
                        )
                        self._replace(
                            module,
                            "forward",
                            MethodType(
                                self._linear(
                                    module.forward,
                                    weight,
                                    bias,
                                    cached_weight,
                                    cached_bias,
                                ),
                                module,
                            ),
                        )
                        self.linear_modules += 1
                    if (
                        isinstance(module, torch.nn.Conv1d)
                        and torch.nn.utils.parametrize.is_parametrized(module, "weight")
                        and getattr(module.forward, "__func__", None)
                        is torch.nn.Conv1d.forward
                    ):
                        weight = module.weight.detach()
                        parameters = tuple(module.parametrizations.weight.parameters())
                        self._replace(
                            module,
                            "forward",
                            MethodType(
                                self._conv(module.forward, weight, parameters), module
                            ),
                        )
                        self.cached_bytes += weight.numel() * weight.element_size()
                        self.conv_modules += 1
                    if (
                        self.quantizer
                        and type(module).__name__ == "MossAudioTokenizerLFQ"
                        and getattr(module.decode_latents, "__func__", None)
                        is type(module).decode_latents
                    ):
                        weight = module.codebook.weight
                        book = F.normalize(weight.float())
                        norm = book.pow(2).sum(1, keepdim=True).T
                        self._replace(
                            module,
                            "decode_latents",
                            MethodType(
                                self._decode(module.decode_latents, weight, book, norm),
                                module,
                            ),
                        )
                        self.cached_bytes += book.numel() * 4 + norm.numel() * 4
                        self.quantizer_modules += 1
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.active:
            return
        try:
            torch.cuda.synchronize(self.device)
        finally:
            storage_epoch.advance(self.device, context=True)
            for module, name, existed, old in reversed(self.saved):
                if existed:
                    setattr(module, name, old)
                else:
                    delattr(module, name)
            self.saved.clear()
            self.warmed.clear()
            self.frequencies.clear()
            self.model.__dict__.pop("_fast_v2_runtime", None)
            self.active = False

    def __exit__(self, *exc):
        self.close()


@contextmanager
def optimized(model, *, cache_linear=True, quantizer=True, pointwise=True):
    """Cache native casts and fuse exact quantizer, rotary and residual operations."""
    with V2Runtime(
        model, cache_linear=cache_linear, quantizer=quantizer, pointwise=pointwise
    ) as runtime:
        yield runtime

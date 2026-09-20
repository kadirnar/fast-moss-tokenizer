"""Reversible optimizations for the pinned 48 kHz stereo v2 checkpoint."""
from contextlib import contextmanager
from threading import get_ident
from types import MethodType
import sys

import torch
import torch.nn.functional as F
from torch.nn.modules import module as module_hooks

from . import storage_epoch
from .v2_loading import MODEL_ID, REVISION, load_model

_LINEAR_FORWARD = torch.nn.Linear.forward


def validate(model):
    if (type(model).__name__ != 'MossAudioTokenizerModel'
            or getattr(model.config,'_commit_hash',None) != REVISION
            or model.sampling_rate != 48000 or model.number_channels != 2
            or model.downsample_rate != 3840 or not model.enable_channel_interleave):
        raise ValueError('Expected the pinned 48 kHz stereo MOSS v2 checkpoint')
    device = next(model.parameters()).device
    if device.type != 'cuda' or torch.__version__ != '2.8.0+cu128':
        raise ValueError('V2 optimizations require CUDA and the validated PyTorch 2.8.0+cu128')
    props = torch.cuda.get_device_properties(device)
    if (props.name != 'NVIDIA GeForce RTX 5070 Ti' or (props.major,props.minor) != (12,0)
            or props.multi_processor_count != 70):
        raise ValueError('V2 kernels require the validated RTX 5070 Ti')
    if model.attention_implementation != 'sdpa':
        raise ValueError('The validated v2 attention backend is SDPA')


class V2Runtime:
    """Single-host-thread owner; original FP32 parameters remain in place."""

    def __init__(self,model,*,cache_linear=True,quantizer=True):
        validate(model)
        self.model = model
        self.device = next(model.parameters()).device
        self.thread = get_ident()
        self.cache_linear = cache_linear
        self.quantizer = quantizer
        self.active = self.used = False
        self.saved = []
        self.warmed = set()
        self.cached_bytes = self.linear_calls = self.prepare_calls = self.select_calls = 0
        self.linear_modules = self.conv_modules = self.quantizer_modules = 0

    def _check(self):
        if not self.active or get_ident() != self.thread:
            raise RuntimeError('Use v2 optimizations on their active owner thread')

    def _replace(self,module,name,value):
        self.saved.append((module,name,name in module.__dict__,module.__dict__.get(name)))
        setattr(module,name,value)

    def _linear(self,original,weight,bias,cached_weight,cached_bias):
        versions = (weight._version, bias._version if bias is not None else None)
        def forward(module,x):
            self._check()
            if (module.weight is not weight or module.bias is not bias
                    or weight._version != versions[0]
                    or (bias is not None and bias._version != versions[1])
                    or x.device != self.device or x.dtype not in (torch.float32,torch.bfloat16)
                    or not torch.is_autocast_enabled('cuda')
                    or torch.get_autocast_dtype('cuda') != torch.bfloat16
                    or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad
                        or (bias is not None and bias.requires_grad)))):
                return original(x)
            self.linear_calls += 1
            # Let the same native autocast/F.linear path cast the activation.
            # The immutable weight cast has already been computed exactly once.
            return F.linear(x,cached_weight,cached_bias)
        return forward

    def _conv(self,original,weight,parameters):
        versions = tuple(p._version for p in parameters)
        def forward(module,x):
            self._check()
            current = tuple(module.parametrizations.weight.parameters()) if torch.nn.utils.parametrize.is_parametrized(module,'weight') else ()
            if (len(current) != len(parameters) or any(a is not b for a,b in zip(current,parameters))
                    or any(p._version != version for p,version in zip(parameters,versions))
                    or module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
                    or any(m._forward_hooks or m._forward_pre_hooks for m in module.parametrizations.modules())
                    or (torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in parameters)))):
                return original(x)
            return module._conv_forward(x,weight,module.bias)
        return forward

    def _decode(self,original,weight,book,norm):
        version = weight._version
        def decode(module,x):
            self._check()
            if (module.codebook.weight is not weight or weight._version != version
                    or x.ndim != 3 or x.shape[1] != 8 or x.device != self.device
                    or x.dtype != torch.float32 or x.shape[0] < 1 or x.shape[2] < 1
                    or torch.is_autocast_enabled('cuda')
                    or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))):
                return original(x)
            if x.is_contiguous() and x.numel() <= 2147483647:
                from .quantizer_prepare import prepare
                stream = torch.cuda.current_stream(self.device).cuda_stream
                key = (id(module),tuple(x.shape),stream)
                if torch.cuda.is_current_stream_capturing() and key not in self.warmed:
                    raise RuntimeError('Warm v2 quantizer preparation on this stream before capture')
                _,row_norm,twice,_ = prepare(x)
                self.warmed.add(key)
                self.prepare_calls += 1
            else:
                enc = F.normalize(x.transpose(1,2).reshape(-1,8).float())
                row_norm = enc.pow(2).sum(1,keepdim=True)
                twice = 2*enc
            from .quantizer import select
            dots = twice @ book.T
            self.select_calls += 1
            return select(dots,row_norm,norm,weight,x)
        return decode

    def __enter__(self):
        if self.used or getattr(self.model,'_fast_v2_runtime',None) is not None or getattr(self.model,'_fast_optimization_active',False):
            raise RuntimeError('V2 optimization context is already active or has been used')
        if get_ident() != self.thread:
            raise RuntimeError('Enter v2 optimization on its construction host thread')
        if (self.model.training or any(p.requires_grad or p.dtype != torch.float32 or p.device != self.device
                                      for p in self.model.parameters())):
            raise ValueError('V2 optimizations require a frozen FP32 checkpoint on one CUDA device')
        if any(getattr(m,'is_streaming',False) for m in self.model.modules()):
            raise RuntimeError('Enter v2 optimization before starting streaming')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Enter v2 optimization before CUDA graph capture')
            torch.cuda.synchronize(self.device)
            storage_epoch.advance(self.device,context=True)
        self.active = self.used = True
        self.model._fast_v2_runtime = self
        try:
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                for module in self.model.modules():
                    if (self.cache_linear and type(module) is torch.nn.Linear
                            and torch.nn.Linear.forward is _LINEAR_FORWARD
                            and getattr(module.forward,'__func__',None) is _LINEAR_FORWARD):
                        weight,bias = module.weight,module.bias
                        cached_weight = weight.to(torch.bfloat16)
                        cached_bias = bias.to(torch.bfloat16) if bias is not None else None
                        self.cached_bytes += cached_weight.numel()*2+(cached_bias.numel()*2 if cached_bias is not None else 0)
                        self._replace(module,'forward',MethodType(self._linear(module.forward,weight,bias,cached_weight,cached_bias),module))
                        self.linear_modules += 1
                    if (isinstance(module,torch.nn.Conv1d) and torch.nn.utils.parametrize.is_parametrized(module,'weight')
                            and getattr(module.forward,'__func__',None) is torch.nn.Conv1d.forward):
                        weight = module.weight.detach()
                        parameters = tuple(module.parametrizations.weight.parameters())
                        self._replace(module,'forward',MethodType(self._conv(module.forward,weight,parameters),module))
                        self.cached_bytes += weight.numel()*weight.element_size()
                        self.conv_modules += 1
                    if (self.quantizer and type(module).__name__=='MossAudioTokenizerLFQ'
                            and getattr(module.decode_latents,'__func__',None) is type(module).decode_latents):
                        weight = module.codebook.weight
                        book = F.normalize(weight.float())
                        norm = book.pow(2).sum(1,keepdim=True).T
                        self._replace(module,'decode_latents',MethodType(self._decode(module.decode_latents,weight,book,norm),module))
                        self.cached_bytes += book.numel()*4+norm.numel()*4
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
            storage_epoch.advance(self.device,context=True)
            for module,name,existed,old in reversed(self.saved):
                if existed:
                    setattr(module,name,old)
                else:
                    delattr(module,name)
            self.saved.clear()
            self.warmed.clear()
            self.model.__dict__.pop('_fast_v2_runtime',None)
            self.active = False

    def __exit__(self,*exc):
        self.close()


@contextmanager
def optimized(model,*,cache_linear=True,quantizer=True):
    """Cache native BF16 linear casts and fuse FP32 quantizer preparation."""
    with V2Runtime(model,cache_linear=cache_linear,quantizer=quantizer) as runtime:
        yield runtime


@torch.no_grad()
def encode_fixed(model,audio):
    """Graph-safe full-frame v2 encoding with unchanged tensor arithmetic.

    Only equal-length complete offline stereo frames are accepted. Their valid
    code length is known from shape, eliminating upstream's CUDA `.item()` sync.
    Use upstream encode/batch_encode for tails or variable per-lane lengths.
    """
    if (audio.ndim != 3 or audio.shape[1] != 2 or audio.shape[0] < 1 or audio.shape[-1] < 3840
            or audio.shape[-1]%3840 or model.sampling_rate != 48000 or model.number_channels != 2
            or model.attention_implementation != 'sdpa'):
        raise ValueError('Expected complete equal-length B,2,T 48 kHz stereo frames with SDPA')
    if any(getattr(m,'is_streaming',False) for m in model.modules()):
        raise RuntimeError('Fixed v2 graphs are offline; use upstream streaming methods for stateful input')
    lengths = torch.full((audio.shape[0],),audio.shape[-1],device=audio.device,dtype=torch.long)
    hidden,lengths = model._flatten_channels_for_codec(audio,lengths)
    with model._codec_inference_autocast():
        for module in model.encoder:
            hidden,lengths = module(hidden,lengths)
    _,codes,code_lengths = model.quantizer(hidden.float(),lengths,None)
    output_type = sys.modules[type(model).__module__].MossAudioTokenizerEncoderOutput
    return output_type(audio_codes=codes,audio_codes_lengths=code_lengths,encoder_hidden_states=hidden.float())


def codec(model,audio):
    """Return owned-by-caller codes, hidden states and stereo reconstruction."""
    enc = encode_fixed(model,audio)
    return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio

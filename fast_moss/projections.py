"""Exact eight-channel LFQ projections and cached decoder reconstruction."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import torch.nn.modules.module as module_hooks
import triton
import triton.language as tl
from . import storage_epoch


@contextmanager
def cache_lifetime(device):
    """Protect externally allocated table pointers used by managed graphs."""
    torch.cuda.synchronize(device)
    storage_epoch.advance(device, context=True)
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        storage_epoch.advance(device, context=True)


@triton.jit
def _project8(X, W, Bias, Y, N: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
              S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, BIAS: tl.constexpr,
              BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, d, t = i // (D * T), i // T % D, i % T
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(8):
        x = tl.load(X + b * S0 + k * S1 + t * S2, i < N, 0)
        w = tl.load(W + d * 8 + k, i < N, 0)
        acc = tl.fma(x, w, acc)
    if BIAS:
        acc = acc + tl.load(Bias + d, i < N, 0)
    tl.store(Y + i, acc, i < N)


@triton.jit
def _project_update(X, W, Bias, Residual, Mask, NewR, Masked,
                    N: tl.constexpr, T: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, d, t = i // (512 * T), i // T % 512, i % T
    acc = tl.full((BLOCK,), 0, tl.float32)
    for k in tl.static_range(8):
        x = tl.load(X + (b * 8 + k) * T + t, i < N, 0)
        w = tl.load(W + d * 8 + k, i < N, 0)
        acc = tl.fma(x, w, acc)
    acc = acc + tl.load(Bias + d, i < N, 0)
    mask = tl.load(Mask + b * T + t, i < N, 0).to(tl.float32)
    update = acc * mask
    residual = tl.load(Residual + i, i < N, 0) - update
    tl.store(NewR + i, residual, i < N)
    tl.store(Masked + i, residual * mask, i < N)


def project_update(x, weight, bias, residual, mask):
    """Internal encoder path: validated canonical FP32 inputs, no observers."""
    out, masked = torch.empty_like(residual), torch.empty_like(residual)
    if residual.numel():
        _project_update[(triton.cdiv(residual.numel(), 256),)](x, weight, bias, residual, mask,
            out, masked, residual.numel(), residual.shape[-1], 256, enable_fp_fusion=False)
    return out, masked


def can_fuse_update(residual, quantizer):
    return (residual.is_cuda and residual.dtype == torch.float32 and residual.is_contiguous()
            and quantizer.out_proj.bias is not None and not torch.is_autocast_enabled('cuda')
            and not torch.backends.cudnn.allow_tf32 and not torch.backends.cudnn.benchmark
            and torch.backends.cudnn.enabled)


def project(x, weight, bias):
    if (x.ndim != 3 or x.shape[1] != 8 or not x.is_cuda or x.dtype != torch.float32
            or weight.shape != (512, 8, 1) or weight.device != x.device or weight.dtype != x.dtype
            or not weight.is_contiguous() or (bias is not None and (bias.shape != (512,)
            or bias.device != x.device or bias.dtype != x.dtype or not bias.is_contiguous()))):
        raise ValueError('Expected CUDA FP32 (B,8,T), contiguous (512,8,1) weight and optional (512,) bias')
    b, _, t = x.shape
    # Singleton-time embedding views can make cuDNN return a different
    # singleton stride (B,512,1) -> (512,1,512). Preserve native dispatch.
    if t == 1 and x.stride() != (8, 1, 1):
        return F.conv1d(x, weight, bias)
    out = torch.empty((b, 512, t), device=x.device, dtype=x.dtype)
    if out.numel():
        _project8[(triton.cdiv(out.numel(), 256),)](x, weight, bias if bias is not None else weight,
            out, out.numel(), 512, t, *x.stride(), bias is not None, 256, enable_fp_fusion=False)
    return out


def forward(self, x):
    if (x.ndim != 3 or x.shape[1] != 8 or x.shape[-1] == 0 or x.shape[0] == 0
            or x.dtype != torch.float32 or x.device != self._fast_weight.device
            or x.requires_grad or torch.is_autocast_enabled('cuda')
            or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark
            or not torch.backends.cudnn.enabled):
        return self._fast_original_projection(x)
    return project(x, self._fast_weight, self.bias)


@triton.jit
def _decode(Codes, Table, Out, N: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
            QUANTIZERS: tl.constexpr, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, d, t = i // (D * T), i // T % D, i % T
    acc = tl.full((BLOCK,), 0, tl.float32)
    for q in range(QUANTIZERS):
        code = tl.load(Codes + q * S0 + b * S1 + t * S2, i < N, 0)
        value = tl.load(Table + (q * SIZE + code) * D + d,
                        (i < N) & (code >= 0) & (code < SIZE), 0)
        acc = acc + value
    tl.store(Out + i, acc, i < N)


@triton.jit
def _decode_tokens(Codes, Table, Out, T: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                   QUANTIZERS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    b, t = token // T, token % T
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for q in range(QUANTIZERS):
        code = tl.load(Codes + q * S0 + b * S1 + t * S2)
        value = tl.load(Table + (q * 1024 + code) * 512 + d,
                        (d < 512) & (code >= 0) & (code < 1024), 0)
        acc = acc + value
    tl.store(Out + (b * 512 + d) * T + t, acc, d < 512)


def reconstruct(codes, table):
    if (codes.ndim != 3 or codes.dtype != torch.long or not codes.is_cuda
            or table.ndim != 3 or table.shape[1:] != (1024, 512) or table.shape[0] != 32
            or table.dtype != torch.float32 or table.device != codes.device or not table.is_contiguous()):
        raise ValueError('Expected CUDA int64 (Q,B,T) codes and contiguous FP32 (32,1024,512) table')
    q, b, t = codes.shape
    q = min(q, table.shape[0])  # Original decode ignores codebooks beyond the model count.
    used = codes[:q]
    # Native embedding lookup raises on invalid indices too. Keep asynchronous
    # GPU validation, with bounded loads even before an assertion is surfaced.
    torch._assert_async(torch.all((used >= 0) & (used < table.shape[1])), 'Invalid decoder code index')
    out = torch.empty((b, 512, t), device=codes.device, dtype=torch.float32)
    if out.numel():
        if t == 1:
            _decode[(triton.cdiv(out.numel(), 256),)](codes, table, out, out.numel(), 512, t,
                *codes.stride(), q, 1024, 256, enable_fp_fusion=False)
        else:
            _decode_tokens[(b * t, 2)](codes, table, out, t, *codes.stride(), q, 256,
                                      enable_fp_fusion=False)
    return out


def decode_codes(self, codes):
    if (codes.ndim != 3 or codes.dtype != torch.long or codes.device != self._fast_projected_codebooks.device
            or torch.is_autocast_enabled('cuda') or torch.backends.cudnn.allow_tf32
            or torch.backends.cudnn.benchmark or not torch.backends.cudnn.enabled
            or module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
            or any(m._forward_hooks or m._forward_pre_hooks for m in self.modules())):
        return self._fast_original_decode_codes(codes)
    return self.output_proj(reconstruct(codes, self._fast_projected_codebooks))


def validate(model):
    device = next(model.parameters()).device
    if device.type != 'cuda':
        raise ValueError('Projection kernels require CUDA')
    properties = torch.cuda.get_device_properties(device)
    if (torch.__version__ != '2.8.0+cu128' or torch.backends.cudnn.version() != 91002
            or properties.name != 'NVIDIA GeForce RTX 5070 Ti' or properties.multi_processor_count != 70):
        raise ValueError('Projection kernels require the validated GPU, PyTorch, and cuDNN versions')
    if torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark or not torch.backends.cudnn.enabled:
        raise ValueError('Projection kernels require enabled cuDNN with TF32 and benchmark disabled')
    q = getattr(model, 'quantizer', None)
    if (type(q).__name__ != 'MossAudioTokenizerResidualLFQ' or q.num_quantizers != 32 or len(q.quantizers) != 32
            or 'decode_codes' in q.__dict__
            or q.codebook_size != 1024 or q.codebook_dim != 8 or q.rvq_dim != 512):
        raise ValueError('Projection kernels require the original 32-codebook LFQ geometry')
    for module in q.quantizers:
        conv = module.out_proj
        if (type(module).__name__ != 'MossAudioTokenizerLFQ' or module.codebook.weight.dtype != torch.float32
                or module.codebook.weight.shape != (1024, 8) or module.codebook.weight.device != device
                or 'decode_code' in module.__dict__ or 'decode_code_wo_out_proj' in module.__dict__
                or not isinstance(conv, torch.nn.Conv1d) or conv.in_channels != 8 or conv.out_channels != 512
                or conv.kernel_size != (1,) or any(p.dtype != torch.float32 or p.device != device for p in conv.parameters())
                or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,)
                or conv.groups != 1 or conv.padding_mode != 'zeros'):
            raise ValueError('Unsupported LFQ projection configuration')

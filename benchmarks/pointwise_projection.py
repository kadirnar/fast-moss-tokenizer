"""Investigate exact FP32 short-channel pointwise projection arithmetic."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from fast_moss.loading import load_model, REVISION
from benchmarks.compare import difference


@triton.jit
def _project(X, W, Bias, Y, N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, T: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
             ORDER: tl.constexpr, BIAS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b, d, t = i // (D * T), i // T % D, i % T
    acc = tl.full((BLOCK,), 0, tl.float32)
    other = tl.full((BLOCK,), 0, tl.float32)
    for j in range(K):
        k = K - 1 - j if ORDER == 1 else j
        x = tl.load(X + b * S0 + k * S1 + t * S2, i < N, 0)
        w = tl.load(W + d * K + k, i < N, 0)
        if ORDER == 2 and j % 2:
            other = tl.fma(x, w, other)
        elif ORDER == 3:
            acc = acc + x * w
        else:
            acc = tl.fma(x, w, acc)
    if ORDER == 2:
        acc = acc + other
    if BIAS:
        acc = acc + tl.load(Bias + d, i < N, 0)
    tl.store(Y + i, acc, i < N)


def project(x, weight, bias, order=0):
    b, k, t = x.shape
    d = weight.shape[0]
    out = torch.empty((b, d, t), device=x.device, dtype=torch.float32)
    _project[(triton.cdiv(out.numel(), 256),)](x, weight, bias if bias is not None else weight,
        out, out.numel(), k, d, t, *x.stride(), order, bias is not None, 256, enable_fp_fusion=False)
    return out


@torch.inference_mode()
def main():
    model = load_model()
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--input-projection', action='store_true')
    args = p.parse_args()
    module = (model.quantizer.quantizers[0].in_proj if args.input_projection else model.quantizer.quantizers[0].out_proj)
    weight = module.weight.detach().contiguous()
    bias = module.bias.detach() if module.bias is not None else None
    print('weight', tuple(weight.shape), 'bias', bias is not None, flush=True)
    records = []
    for batch, frames in [(1, 1), (1, 3), (8, 3), (128, 3), (8, 40)]:
        torch.manual_seed(291)
        x = torch.randn(batch, weight.shape[1], frames, device='cuda') * .2
        ref = F.conv1d(x, weight, bias)
        choices = {f'order_{i}': difference(ref, project(x, weight, bias, i)) for i in range(4)}
        mm = F.linear(x.transpose(1, 2).contiguous(), weight[..., 0], None).transpose(1, 2).contiguous()
        if bias is not None: mm = mm + bias[None, :, None]
        choices['linear'] = difference(ref, mm)
        from triton.testing import do_bench_cudagraph
        timings = {name: do_bench_cudagraph(fn, rep=30, return_mode='median') for name,fn in [('conv',lambda:F.conv1d(x,weight,bias)), ('forward_fma',lambda:project(x,weight,bias,0))]}
        record = {'shape': list(x.shape), 'comparisons': choices, 'graph_ms':timings, 'component_gain':timings['conv']/timings['forward_fma']}
        records.append(record)
        print(record, flush=True)
    Path('results/pointwise_input_probe.json' if args.input_projection else 'results/pointwise_projection_probe.json').write_text(json.dumps({'revision':REVISION,
        'scope':'first actual input projection weight; seeded synthetic activations' if args.input_projection else 'first actual output projection weight; seeded synthetic activations',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cudnn':torch.backends.cudnn.version(),
        'weight_shape':list(weight.shape),'records':records}, indent=2)+'\n')


if __name__ == '__main__': main()

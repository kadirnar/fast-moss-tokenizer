"""Research-only FP32 GEMM with explicit sequential FMA and split boundaries.

This is not enabled by optimized(). Flattened 2-D component equality is not
enough to establish native higher-rank dispatch or full-model fidelity.
"""
import argparse
import ctypes as C
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.testing import do_bench_cudagraph

from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.cublaslt import LinearPlan, I, SZ, check, library
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION, strict_precision


@triton.jit
def _partials(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              CHUNK: tl.constexpr, BN: tl.constexpr, UNROLL: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m, part = tl.program_id(1), tl.program_id(2)
    acc = tl.full((BN,), 0, tl.float32)
    for block in range(tl.cdiv(CHUNK, UNROLL)):
        for step in tl.static_range(UNROLL):
            local = block * UNROLL + step
            k = part * CHUNK + local
            x = tl.load(X + m * K + k, (k < K) & (local < CHUNK), 0)
            w = tl.load(W + k * N + n, (n < N) & (k < K) & (local < CHUNK), 0)
            acc = tl.fma(x, w, acc)
    tl.store(P + (part * M + m) * N + n, acc, n < N)


@triton.jit
def _reduce(P, Y, NUMEL: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.load(P + i, i < NUMEL, 0)
    for part in range(1, PARTS):
        acc = acc + tl.load(P + part * NUMEL + i, i < NUMEL, 0)
    tl.store(Y + i, acc, i < NUMEL)


def ordered(x, packed, chunk, bn=128, unroll=8):
    m, k = x.shape
    n = packed.shape[1]
    parts = triton.cdiv(k, chunk)
    partials = torch.empty((parts, m, n), device=x.device, dtype=torch.float32)
    _partials[(triton.cdiv(n, bn), m, parts)](x, packed, partials, m, n, k,
        chunk, bn, unroll, enable_fp_fusion=False)
    if parts == 1:
        return partials[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _reduce[(triton.cdiv(m*n, 256),)](partials, out, m*n, parts, 256,
                                     enable_fp_fusion=False)
    return out


@triton.jit
def _tiled(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           CHUNK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(tl.cdiv(CHUNK, BK)):
        local = block * BK + kk
        k = part * CHUNK + local
        x = tl.load(X + m[:, None] * K + k[None, :],
                    (m[:, None] < M) & (k[None, :] < K) & (local[None, :] < CHUNK), 0)
        w = tl.load(W + k[:, None] * N + n[None, :],
                    (n[None, :] < N) & (k[:, None] < K) & (local[:, None] < CHUNK), 0)
        acc = tl.dot(x, w, acc, input_precision='ieee')
    tl.store(P + (part * M + m[:, None]) * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def tiled(x, packed, chunk, tile=(16, 32, 32, 4), return_kernel=False):
    m, k = x.shape
    n = packed.shape[1]
    parts = triton.cdiv(k, chunk)
    partials = torch.empty((parts, m, n), device=x.device, dtype=torch.float32)
    bm, bn, bk, warps = tile
    kernel = _tiled[(triton.cdiv(m, bm), triton.cdiv(n, bn), parts)](
        x, packed, partials, m, n, k, chunk, bm, bn, bk, num_warps=warps,
        enable_fp_fusion=False)
    if parts == 1:
        out = partials[0]
    else:
        out = torch.empty((m, n), device=x.device, dtype=torch.float32)
        _reduce[(triton.cdiv(m*n, 256),)](partials, out, m*n, parts, 256,
                                         enable_fp_fusion=False)
    return (out, kernel) if return_kernel else out


def configuration(plan, index):
    config = {}
    for attr, name in enumerate(['id', 'tile', 'split_k', 'reduction', 'swizzle', 'custom', 'stage']):
        value, size = I(), SZ()
        check(plan.lib.cublasLtMatmulAlgoConfigGetAttribute(C.byref(plan.algorithms[index].algo),
            attr, C.byref(value), C.sizeof(value), C.byref(size)))
        config[name] = value.value
    return config


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', default='results/matrix_inputs.pt')
    p.add_argument('--rows', type=int, nargs='+', default=[3, 24])
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--output', default='results/ordered_mm.json')
    p.add_argument('--tiled', action='store_true')
    a = p.parse_args()
    strict_precision()
    cases = torch.load(a.inputs, weights_only=True)
    profile = json.loads(Path('fast_moss/matrix_profile.json').read_text())
    selected = {tuple(r['shape']): r for r in profile['records']}
    items = sorted(((s,c) for s,c in cases.items() if s[0] in a.rows),
                   key=lambda v: v[1]['calls'] * v[0][1] * v[0][2], reverse=True)
    if a.limit:
        items = items[:a.limit]
    report = {'scope': 'research-only ordered FP32 component GEMM, not model speedup',
              'revision': REVISION, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'cublaslt_version': library().cublasLtGetVersion(), 'tf32': False,
              'reference': 'contiguous flattened 2-D F.linear', 'records': []}
    flush = torch.empty(128*1024*1024, device='cuda', dtype=torch.uint8)
    for shape, case in items:
        x, w = case['x'], case['weight']
        packed = w.T.contiguous()
        ref = F.linear(x, w)
        r = {'shape': shape, 'layer': case['name'], 'trials': []}
        chunk_sizes = {8, 16, 32, 64, 128, 256, 512, shape[-1]}
        with LinearPlan(w, shape[0], 'packed', packed_weight=packed) as plan:
            if shape in selected:
                idx = plan.restore(selected[shape]['algorithm'], profile['cublaslt_version'])
                r['configuration'] = configuration(plan, idx)
                r['selected_vs_reference'] = difference(ref, plan(x, idx))
                splits = max(1, r['configuration']['split_k'])
                for alignment in [8, 16, 32]:
                    chunk_sizes.add(triton.cdiv(triton.cdiv(shape[-1], splits), alignment) * alignment)
                baseline = lambda z: plan(z, idx)
            else:
                baseline = lambda z: F.linear(z, w)
            if a.tiled:
                chunk_sizes = {256}
            for chunk in sorted(chunk_sizes):
                out = ordered(x, packed, chunk)
                trial = {'chunk': chunk, 'difference': difference(ref, out)}
                r['trials'].append(trial)
                print(shape, chunk, trial['difference'], flush=True)
                if trial['difference']['exact']:
                    graph = GraphedCallable(lambda z: (ordered(z, packed, chunk),), x)
                    trial['graph_vs_reference'] = difference(ref, graph(x)[0])
                    trial['warm_ms'] = do_bench_cudagraph(lambda: ordered(x, packed, chunk), rep=20)
                    trial['cold'] = evicted_replay(graph, flush)
                    del graph
            if a.tiled:
                r['tiled'] = []
                for bm in [16, 32]:
                    for bn in [32, 64, 128]:
                        for bk in [32, 64]:
                            tile = (bm, bn, bk, 4)
                            out, kernel = tiled(x, packed, 256, tile, return_kernel=True)
                            trial = {'tile': tile, 'chunk': 256, 'difference': difference(ref, out),
                                     'registers': kernel.n_regs, 'spills': kernel.n_spills}
                            if trial['difference']['exact']:
                                trial['warm_ms'] = do_bench_cudagraph(lambda: tiled(x, packed, 256, tile), rep=15)
                            r['tiled'].append(trial)
                            print(shape, tile, trial, flush=True)
            graph = GraphedCallable(lambda z: (baseline(z),), x)
            r['baseline'] = {'warm_ms': do_bench_cudagraph(lambda: baseline(x), rep=20),
                             'cold': evicted_replay(graph, flush)}
            del graph
        report['records'].append(r)
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

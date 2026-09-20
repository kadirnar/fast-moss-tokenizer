"""Actual-weight compression ratios, decode cost, and exact vendor GEMM timing."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.lossless_weights import pack
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.loading import strict_precision, REVISION
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/lossless_shapes.json')
    p.add_argument('--limit', type=int, default=12)
    a = p.parse_args(); strict_precision()
    cases = torch.load('results/matrix_inputs.pt', weights_only=True)
    cases = sorted(((s, c) for s, c in cases.items() if s[0] in [1, 3, 24, 384]),
                   key=lambda item: item[1]['calls'] * item[0][1] * item[0][2], reverse=True)[:a.limit]
    report = {'scope': 'lossless FP32 storage plus separate reconstruction and unchanged vendor GEMM; research only',
              'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'storage': '24 sign/mantissa bits plus block exponent deltas; integer bitstream',
              'arithmetic': 'original FP32 F.linear', 'tf32': False, 'records': []}
    flush = torch.empty(128 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    for shape, case in cases:
        x, w = case['x'], case['weight']
        scratch = torch.empty_like(w)
        reference = F.linear(x, w)
        def timing(fn):
            graph = GraphedCallable(lambda z: (fn(z),), x)
            data = {'hot_graph_ms': do_bench_cudagraph(lambda: fn(x), rep=20, return_mode='median'),
                    'evicted': evicted_replay(graph, flush), 'graph_exact': torch.equal(reference, graph(x)[0])}
            del graph
            return data
        baseline = timing(lambda z: F.linear(z, w))
        def copied(z):
            scratch.copy_(w)
            return F.linear(z, scratch)
        copy_timing = timing(copied)
        record = {'shape_MNK': shape, 'layer': case['name'], 'baseline': baseline,
                  'copy_then_gemm': copy_timing, 'variants': []}
        for block in [128, 256, 512, 1024]:
            compressed = pack(w, block)
            restored = compressed.unpack(scratch)
            exact_bits = torch.equal(w.view(torch.int32), restored.view(torch.int32))
            def restored_linear(z):
                return F.linear(z, compressed.unpack(scratch))
            candidate = restored_linear(x)
            result = timing(restored_linear)
            unpack_ms = do_bench_cudagraph(lambda: compressed.unpack(scratch), rep=20, return_mode='median')
            v = {'block': block, 'weight_bits_exact': exact_bits, 'matrix': difference(reference, candidate),
                 'original_bytes': w.numel() * w.element_size(), 'compressed_bytes': compressed.nbytes,
                 'ratio': w.numel() * w.element_size() / compressed.nbytes,
                 'exponent_bits_histogram': torch.bincount((compressed.headers >> 8).long(), minlength=9).tolist(),
                 'unpack_hot_ms': unpack_ms, 'timing': result,
                 'hot_gain': baseline['hot_graph_ms'] / result['hot_graph_ms'],
                 'evicted_gain': baseline['evicted']['gpu_ms_median'] / result['evicted']['gpu_ms_median']}
            record['variants'].append(v)
            print(shape, block, 'ratio', round(v['ratio'], 3), 'warm/cold gains',
                  round(v['hot_gain'], 3), round(v['evicted_gain'], 3), 'exact', exact_bits and v['matrix']['exact'], flush=True)
        record['baseline_after'] = timing(lambda z: F.linear(z, w))
        report['records'].append(record)
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    report['all_exact'] = all(v['weight_bits_exact'] and v['matrix']['exact'] and v['timing']['graph_exact']
                              for r in report['records'] for v in r['variants'])
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']: raise SystemExit('Lossless storage/GEMM gate failed')


if __name__ == '__main__': main()

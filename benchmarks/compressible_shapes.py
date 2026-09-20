"""Transparent hardware compression versus ordinary allocations and positive controls."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.compressible_memory import Allocation
from benchmarks.matrices import evicted_replay
from benchmarks.compare import difference
from fast_moss.loading import strict_precision, REVISION
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/compressible_shapes.json')
    p.add_argument('--limit', type=int, default=12)
    a = p.parse_args(); strict_precision()
    cases = torch.load('results/matrix_inputs.pt', weights_only=True)
    selected = sorted(((s, c) for s, c in cases.items() if s[0] in [1, 3, 24, 384]),
                      key=lambda item: item[1]['calls'] * item[0][1] * item[0][2], reverse=True)[:a.limit]
    controls = []
    base_shape, base_case = selected[0]
    for name, w in [('zeros', torch.zeros_like(base_case['weight'])),
                    ('bf16_roundtrip_control', base_case['weight'].bfloat16().float())]:
        controls.append((base_shape, {'name': name, 'x': base_case['x'], 'weight': w}))
    report = {'scope': 'original FP32 weights in verified CUDA generic-compression allocations; research only',
              'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'arithmetic': 'unchanged FP32 F.linear', 'tf32': False,
              'controls': 'Zero and BF16-roundtrip matrices only test hardware compressibility; they are not codec replacements',
              'records': []}
    flush = torch.empty(128 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    for shape, case in selected + controls:
        x, original = case['x'], case['weight']
        owners = {name: Allocation(tuple(original.shape), compressed) for name, compressed in [('vmm_plain', False), ('vmm_compressed', True)]}
        weights = {'torch': original, **{name: owner.tensor() for name, owner in owners.items()}}
        for name in owners: weights[name].copy_(original)
        reference = F.linear(x, original)
        record = {'shape_MNK': shape, 'layer': case['name'], 'is_control': case['name'] in ['zeros', 'bf16_roundtrip_control'],
                  'allocations': {name: owner.metadata() for name, owner in owners.items()},
                  'weight_bits_exact': all(torch.equal(w.view(torch.int32), original.view(torch.int32)) for w in weights.values()),
                  'rounds': []}
        for repeat in range(3):
            order = list(weights) if repeat % 2 == 0 else list(reversed(weights))
            for name in order:
                w = weights[name]; fn = lambda z: (F.linear(z, w),)
                graph = GraphedCallable(fn, x)
                result = {'round': repeat, 'allocation': name, 'fidelity': difference(reference, graph(x)[0]),
                          'hot_ms': do_bench_cudagraph(lambda: F.linear(x, w), rep=20, return_mode='median'),
                          'evicted': evicted_replay(graph, flush)}
                del graph
                record['rounds'].append(result)
        record['medians'] = {name: {'hot_ms': statistics.median(r['hot_ms'] for r in record['rounds'] if r['allocation'] == name),
                                     'evicted_ms': statistics.median(r['evicted']['gpu_ms_median'] for r in record['rounds'] if r['allocation'] == name)} for name in weights}
        record['cold_gain_vs_torch'] = record['medians']['torch']['evicted_ms'] / record['medians']['vmm_compressed']['evicted_ms']
        record['cold_gain_vs_plain_vmm'] = record['medians']['vmm_plain']['evicted_ms'] / record['medians']['vmm_compressed']['evicted_ms']
        record['all_exact'] = record['weight_bits_exact'] and all(r['fidelity']['exact'] for r in record['rounds'])
        report['records'].append(record)
        print(shape, case['name'], 'cold gains torch/plain', round(record['cold_gain_vs_torch'], 3), round(record['cold_gain_vs_plain_vmm'], 3), 'exact', record['all_exact'], flush=True)
        # Functions close over w; release tensors before allocator owners.
        del fn, w, weights, owners
        gc.collect()
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    report['all_exact'] = all(r['all_exact'] for r in report['records'])
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']: raise SystemExit('VMM compression fidelity failed')


if __name__ == '__main__': main()

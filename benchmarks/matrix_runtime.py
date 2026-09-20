"""Paired full-checkpoint ablation of the supported resident matrix backend."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='results/full_matrix_runtime.json')
    args = parser.parse_args()
    model = load_model()
    clips, sources = audio_sources()
    report = {'scope': 'full checkpoint, supported matrix backend against previous optimized runtime',
              'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'dtype': 'float32', 'tf32': False, 'quantizers': 32, 'sources': sources,
              'packing': 'lazy, only weights reached by a validated matrix shape; synchronize storage transitions',
              'profile': json.loads(Path('fast_moss/matrix_profile.json').read_text()),
              'reference': 'original eager model for each source and geometry',
              'timing_scope': 'warmed combined encode/decode graph with input/output copies; excludes packing, plan construction, capture, and restoration; setup and peak allocated memory recorded separately',
              'input': 'cyclic source samples, lanes offset by 1920 samples', 'cases': []}
    def run(x):
        enc = model._encode_frame(x)
        return enc.audio_codes, enc.encoder_hidden_states, model._decode_frame(enc.audio_codes).audio
    labels = ['codes', 'hidden', 'audio']
    for batch, frames in [(1, 1), (1, 3), (8, 3), (128, 3)]:
        idx = torch.arange(frames * 1920, device='cuda')[None] + torch.arange(batch, device='cuda')[:, None] * 1920
        inputs = [clip[idx % clip.numel()][:, None] for clip in clips]
        refs = [tuple(v.cpu() for v in run(x)) for x in inputs]
        rounds = {'none': [], 'cublaslt': []}
        for repeat in range(args.repeats):
            modes = list(rounds) if repeat % 2 == 0 else list(reversed(rounds))
            for mode in modes:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                with optimized(model, residual_backend='triton', kv_backend='triton', rope_backend='triton',
                               share_rope_tables=True, attention_mask_backend='triton',
                               quantizer_backend='triton', matrix_backend=mode):
                    records = []
                    for i, (x, ref) in enumerate(zip(inputs, refs)):
                        eager = run(x)
                        graph = GraphedCallable(run, x)
                        replay = graph(x)
                        checks = {name: {label: difference(old, new.cpu()) for label, old, new in zip(labels, ref, values)}
                                  for name, values in [('eager', eager), ('graph', replay)]}
                        records.append({'source': sources[i]['path'], 'comparisons': checks})
                        if i == 0:
                            torch.cuda.synchronize()
                            setup_seconds = time.perf_counter() - before
                            timing = measure(lambda: graph(x), repeats=20)
                        del graph, eager, replay
                    runtime = getattr(model, '_fast_matrix_runtime', None)
                    resources = ({'plan_count': len(runtime.plans), 'workspace_count': len(runtime.workspaces),
                                  'workspace_bytes': sum(w.numel() for w in runtime.workspaces.values()),
                                  'packed_weight_bytes': runtime.packed_bytes} if runtime else {})
                    del runtime
                restored = run(inputs[0])
                restoration = {label: difference(old, new.cpu()) for label, old, new in zip(labels, refs[0], restored)}
                del restored
                record = {'round': repeat, 'timing': timing, 'setup_seconds': setup_seconds,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'resources': resources,
                          'sources': records, 'restored': restoration}
                record['all_exact'] = (all(v['exact'] for r in records for c in r['comparisons'].values() for v in c.values())
                                       and all(v['exact'] for v in restoration.values()))
                rounds[mode].append(record)
                print(batch, frames, repeat, mode, round(timing['wall_ms_median'], 3),
                      'ms', 'exact', record['all_exact'], flush=True)
        medians = {mode: statistics.median(r['timing']['wall_ms_median'] for r in entries)
                   for mode, entries in rounds.items()}
        case = {'batch': batch, 'frames': frames, 'rounds': rounds, 'wall_ms_medians': medians,
                'speedup': medians['none'] / medians['cublaslt'],
                'all_exact': all(r['all_exact'] for entries in rounds.values() for r in entries)}
        report['cases'].append(case)
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        del refs, inputs
    report['all_exact'] = all(c['all_exact'] for c in report['cases'])
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']:
        raise SystemExit('Supported matrix runtime fidelity failed')


if __name__ == '__main__':
    main()

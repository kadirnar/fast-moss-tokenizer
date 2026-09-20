"""Full-checkpoint incremental arrivals: exactness and matched fragmentation cost."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from benchmarks.request_batching import reference
from fast_moss.batching import StreamingBatcher
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized


@torch.inference_mode()
def arrivals(batcher, inputs, fragmented):
    ids = [batcher.submit(final=False) for _ in inputs]
    indices = {key: i for i, key in enumerate(ids)}
    positions = [0] * len(inputs)
    finished = [False] * len(inputs)
    output = [[] for _ in inputs]
    offsets, lanes, first, last_input, finals = {}, {}, {}, {}, set()
    tick = steps = idle = 0
    peak = 0
    while batcher.pending:
        for i, value in enumerate(inputs):
            if finished[i] or (tick + i) % 4 == 0 or (i == 0 and tick < 3):
                continue
            start = positions[i]
            if start == value.shape[-1]:
                # Late final notification, after all bytes were previously sent.
                batcher.append(ids[i], final=True)
                finished[i] = True
                continue
            end = min(start + batcher.width, value.shape[-1])
            if fragmented and end - start > 2:
                cuts = [start, start + 1, start + max(2, (end - start) // 2), end]
            else:
                cuts = [start, end]
            for left, right in zip(cuts, cuts[1:]):
                batcher.append(ids[i], value[..., left:right])
            positions[i] = end
            if end == value.shape[-1]:
                last_input[i] = tick
        peak = max(peak, batcher.buffered_bytes)
        ready = batcher.can_step
        chunks = batcher.step()
        if not ready:
            assert not chunks
            idle += 1
        if any(c.input_length for c in chunks):
            steps += 1
        for chunk in chunks:
            i = indices[chunk.request_id]
            assert chunk.offset == offsets.get(i, 0)
            assert chunk.lane == lanes.setdefault(i, chunk.lane)
            assert i not in finals
            output[i].append(chunk.data)
            offsets[i] = chunk.offset + chunk.data.shape[-1]
            if chunk.input_length:
                first.setdefault(i, tick)
            if chunk.final:
                finals.add(i)
        tick += 1
        if tick > 1000:
            raise AssertionError('Incremental schedule failed to drain')
    assert finals == set(range(len(inputs))) and batcher.buffered_bytes == 0
    return [torch.cat(parts, -1) for parts in output], {
        'logical_ticks': tick, 'model_steps': steps, 'idle_steps': idle,
        'lanes': lanes, 'first_output_tick': first, 'last_input_tick': last_input,
        'requests_output_before_last_input': [i for i in first if first[i] < last_input[i]],
        'peak_retained_input_bytes': peak,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['triton', 'cute'], default='triton')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--matrix-backend', choices=['none', 'cublaslt'], default='none')
    parser.add_argument('--output', default='results/full_incremental_batching.json')
    parser.add_argument("--projection-backend", choices=["none", "triton"], default="none")
    args = parser.parse_args()
    model = load_model()
    clips, sources = audio_sources()
    batch, frames = 8, 3
    counts = [129, 3, 6, 9, 12, 15, 18, 21, 129, 4]
    audio = []
    for i, count in enumerate(counts):
        source = clips[i % len(clips)]
        samples = count * model.downsample_rate - (17 if i % 3 else 0)
        indices = (torch.arange(samples, device='cuda') + (i % batch) * 1920) % source.numel()
        audio.append(source[indices][None])
    encoded = reference(model, 'encode', audio, batch, frames)
    report = {
        'scope': 'full checkpoint; incremental input, original FP32 weights and arithmetic',
        'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'dtype': 'float32', 'tf32': False, 'quantizers': 32, 'batch': batch,
        'chunk_frames': frames, 'backend': args.backend, 'quantizer_backend': 'triton', 'matrix_backend': args.matrix_backend, 'projection_backend': args.projection_backend,
        'sources': sources, 'audio_input_samples': [x.shape[-1] for x in audio],
        'input': 'source i%3 repeated cyclically, starting at (i%8)*1920 samples',
        'reference': 'independent original eager timelines at the same batch shape',
        'arrival_schedule': 'All requests open empty. Each tick appends up to one chunk per request, except (tick+i)%4==0; request zero waits until tick three. Final notification follows the last data arrival on the next eligible tick. One step per tick.',
        'comparison': 'Identical logical arrivals: one fragment versus three nonempty fragments (1, max(2,floor(n/2))-1, remaining) for arrivals longer than two elements. Both use the current optimized scheduler.',
        'timing_scope': 'Warmed graph, host-driven loop including input clones, metadata uploads, gather, resets, output ownership and concatenation; excludes model load/capture/references. Ticks have no real-time waits and do not model network latency.',
        'results': {},
    }
    for direction, inputs in [('encode', audio), ('decode', encoded)]:
        refs = encoded if direction == 'encode' else reference(model, direction, inputs, batch, frames)
        rounds = {'one_fragment': [], 'three_fragments': []}
        with optimized(model, residual_backend=args.backend, kv_backend='triton', rope_backend='triton',
                       share_rope_tables=True, attention_mask_backend='triton', quantizer_backend='triton',
                       matrix_backend=args.matrix_backend, projection_backend=args.projection_backend):
            with StreamingBatcher(model, direction, batch, frames) as batcher:
                # Warm the complete schedule, including all metadata shapes and
                # tails. A short prefix misses some gather specializations.
                arrivals(batcher, inputs, False)
                arrivals(batcher, inputs, True)
                graph = batcher.session.graph
                for repeat in range(args.repeats):
                    modes = list(rounds) if repeat % 2 == 0 else list(reversed(rounds))
                    for mode in modes:
                        torch.cuda.synchronize()
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        before = time.perf_counter()
                        start.record()
                        values, stats = arrivals(batcher, inputs, mode == 'three_fragments')
                        end.record(); end.synchronize()
                        wall_ms = (time.perf_counter() - before) * 1000
                        checks = [difference(ref, value) for ref, value in zip(refs, values)]
                        entry = {**stats, 'wall_ms': wall_ms, 'gpu_ms': start.elapsed_time(end),
                                 'same_graph': graph is batcher.session.graph, 'fidelity': checks}
                        rounds[mode].append(entry)
                        print(direction, repeat, mode, round(wall_ms, 3), 'ms',
                              'exact', all(c['exact'] for c in checks), flush=True)
        medians = {mode: statistics.median(r['wall_ms'] for r in entries) for mode, entries in rounds.items()}
        result = {'rounds': rounds, 'wall_ms_medians': medians,
                  'fragmented_overhead_percent': 100 * (medians['three_fragments'] / medians['one_fragment'] - 1),
                  'all_exact': all(r['same_graph'] and all(c['exact'] for c in r['fidelity'])
                                   for entries in rounds.values() for r in entries)}
        report['results'][direction] = result
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    report['all_exact'] = all(r['all_exact'] for r in report['results'].values())
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']:
        raise SystemExit('Incremental batching fidelity failed')


if __name__ == '__main__':
    main()

"""Full-checkpoint FIFO refill fidelity and matched fixed-wave throughput."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from fast_moss.batching import StreamingBatcher
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.streaming import StreamingSession


@torch.inference_mode()
def reference(model, direction, inputs, batch, frames):
    """Independent original eager timelines, retaining the scheduled batch shape."""
    result = []
    width = frames * (model.downsample_rate if direction == 'encode' else 1)
    for index, value in enumerate(inputs):
        chunks = []
        with StreamingSession(model, direction, batch, frames, use_graph=False, fast_reset=False) as session:
            for start in range(0, value.shape[-1], width):
                part = value[..., start:start + width]
                x = (part[None].expand(batch, -1, -1).contiguous() if direction == 'encode'
                     else part[:, None].expand(-1, batch, -1).contiguous())
                out, _ = session.push(x, final=start + width >= value.shape[-1])
                chunks.append((out[:, 0] if direction == 'encode' else out[0]).clone())
        result.append(torch.cat(chunks, -1))
        print(direction, 'reference request', index, flush=True)
    return result


@torch.inference_mode()
def run_queue(batcher, inputs, wave):
    results = [[] for _ in inputs]
    ids = {}; next_input = 0; steps = 0; active_slots = 0; ownership = {}
    while next_input < len(inputs) or batcher.pending:
        if not wave or not batcher.pending:
            end = min(next_input + batcher.batch_size, len(inputs)) if wave else len(inputs)
            for i in range(next_input, end):
                ids[batcher.submit(inputs[i])] = i
            next_input = end
        for chunk in batcher.step():
            i = ids[chunk.request_id]
            if ownership.setdefault(i, chunk.lane) != chunk.lane:
                raise AssertionError('A request changed batch slots')
            results[i].append(chunk.data)
            active_slots += 1
        steps += 1
    return [torch.cat(chunks, -1) for chunks in results], {
        'steps': steps, 'active_slots': active_slots,
        'slot_utilization': active_slots / (steps * batcher.batch_size), 'lanes': ownership}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--backend', choices=['triton', 'cute'], default='triton')
    p.add_argument('--output', default='results/full_request_batching.json')
    p.add_argument("--quantizer-backend", choices=["none", "triton"], default="none")
    a = p.parse_args()
    model = load_model(); clips, sources = audio_sources()
    batch, frames = 8, 3
    frame_counts = [129, 3, 6, 9, 12, 15, 18, 21] * 2
    audio = []
    for i, n in enumerate(frame_counts):
        source = clips[i % len(clips)]
        count = n * 1920 - (17 if i % 3 else 0)
        indices = (torch.arange(count, device='cuda') + (i % batch) * 1920) % source.numel()
        audio.append(source[indices][None])
    report = {'scope': 'full-checkpoint finite-request scheduling, original FP32 weights/arithmetic',
              'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'dtype': 'float32', 'tf32': False, 'quantizers': 32, 'batch': batch,
              'chunk_frames': frames, 'backend': a.backend, 'quantizer_backend': a.quantizer_backend, 'sources': sources,
              'input': 'source i%3 repeated cyclically, starting at (i%8)*1920 samples',
              'audio_input_samples': [x.shape[-1] for x in audio],
              'reference': 'independent original eager timelines with the same batch shape',
              'timing_scope': 'warmed graph; includes submit copies, gather, lane reset, output copies and concatenation; excludes model loading/capture/reference construction',
              'baseline': 'same scheduler and batch shape, new groups admitted only after the previous eight requests finish',
              'results': {}}
    encoded = reference(model, 'encode', audio, batch, frames)
    for direction, inputs in [('encode', audio), ('decode', encoded)]:
        refs = encoded if direction == 'encode' else reference(model, direction, inputs, batch, frames)
        audio_seconds = sum(x.shape[-1] for x in audio) / model.sampling_rate
        rounds = {'fifo': [], 'fixed_waves': []}
        with optimized(model, residual_backend=a.backend, kv_backend='triton', rope_backend='triton',
                       share_rope_tables=True, attention_mask_backend='triton', quantizer_backend=a.quantizer_backend):
            with StreamingBatcher(model, direction, batch, frames) as batcher:
                # Compile and capture once. A completed warm request also tests
                # that resetting the next occupant preserves graph addresses.
                width = frames * (1920 if direction == 'encode' else 1)
                for _ in range(batch): batcher.submit(inputs[0][..., :width])
                batcher.step(); captured = batcher.session.graph
                for repeat in range(a.repeats):
                    modes = ['fifo', 'fixed_waves'] if repeat % 2 == 0 else ['fixed_waves', 'fifo']
                    for mode in modes:
                        torch.cuda.synchronize()
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        before = time.perf_counter(); start.record()
                        out, stats = run_queue(batcher, inputs, mode == 'fixed_waves')
                        end.record(); end.synchronize()
                        wall_ms = (time.perf_counter() - before) * 1000
                        checks = [difference(ref, value) for ref, value in zip(refs, out)]
                        entry = {**stats, 'wall_ms': wall_ms, 'gpu_ms': start.elapsed_time(end),
                                 'same_graph': batcher.session.graph is captured, 'fidelity': checks,
                                 'audio_seconds_per_second': audio_seconds / (wall_ms / 1000)}
                        rounds[mode].append(entry)
                        print(direction, repeat, mode, round(wall_ms, 2), 'ms', stats['steps'], 'steps',
                              'exact', all(c['exact'] for c in checks), flush=True)
        result = {'rounds': rounds, 'audio_seconds': audio_seconds,
                  'all_exact': all(r['same_graph'] and all(c['exact'] for c in r['fidelity'])
                                   for entries in rounds.values() for r in entries)}
        result['wall_ms_medians'] = {name: statistics.median(r['wall_ms'] for r in entries)
                                     for name, entries in rounds.items()}
        result['speedup_over_fixed_waves'] = result['wall_ms_medians']['fixed_waves'] / result['wall_ms_medians']['fifo']
        report['results'][direction] = result
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    report['all_exact'] = all(r['all_exact'] for r in report['results'].values())
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']: raise SystemExit('Request batching fidelity failed')


if __name__ == '__main__': main()

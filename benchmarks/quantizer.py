"""Full-checkpoint quantizer-stage and encoder ablation at identical batch shapes."""
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/full_quantizer.json')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--select-only', action='store_true', help='Reproduce the initial selector/embedding/STE-only variant')
    a = p.parse_args()
    model = load_model(); clips, sources = audio_sources()
    report = {'scope': 'full checkpoint exact quantizer fusion; baseline is previous optimized runtime',
              'revision': REVISION, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'dtype': 'float32', 'tf32': False, 'quantizers': 32, 'sources': sources,
              'quantizer_variant': 'select_only' if a.select_only else 'full',
              'input': 'cyclic speech, lane offsets of 1920 samples', 'cases': []}
    for batch, frames in [(1, 1), (1, 3), (8, 3), (128, 3), (8, 40)]:
        idx = torch.arange(frames * 1920, device='cuda')[None] + torch.arange(batch, device='cuda')[:, None] * 1920
        x = clips[1][idx % clips[1].numel()][:, None]
        captured = []
        handle = model.quantizer.register_forward_pre_hook(lambda m, args: captured.append(tuple(v.clone() if isinstance(v, torch.Tensor) else v for v in args)))
        ref = model._encode_frame(x)
        handle.remove()
        z, lengths, _ = captured[0]
        qref = model.quantizer(z, lengths)
        record = {'batch': batch, 'frames': frames, 'quantizer_input_shape': list(z.shape),
                  'quantizer_input_stride': list(z.stride()), 'rounds': []}
        original_q_forward = model.quantizer.forward
        for repeat in range(a.rounds):
            for backend in (['none', 'triton'] if repeat % 2 == 0 else ['triton', 'none']):
                with optimized(model, residual_backend='triton', kv_backend='triton', rope_backend='triton',
                               share_rope_tables=True, attention_mask_backend='triton', quantizer_backend=backend):
                    if a.select_only and backend == 'triton':
                        model.quantizer.forward = original_q_forward
                        model._encode_frame = model._fast_original_encode_frame
                    fn = lambda audio: (model._encode_frame(audio).audio_codes,)
                    qfn = lambda hidden, n: model.quantizer(hidden, n)
                    eager = qfn(z, lengths)
                    graph = GraphedCallable(qfn, z, lengths)
                    out = graph(z, lengths)
                    qchecks = {mode: [difference(r, c) for r, c in zip(qref, values)]
                               for mode, values in [('eager', eager), ('graph', out)]}
                    qtime = measure(lambda: graph(z, lengths), repeats=20)
                    del graph
                    graph = GraphedCallable(fn, x)
                    codes = graph(x)[0]
                    etime = measure(lambda: graph(x), repeats=20)
                    del graph
                    record['rounds'].append({'round': repeat, 'backend': backend,
                                             'quantizer_fidelity': qchecks,
                                             'encoder_fidelity': difference(ref.audio_codes, codes),
                                             'quantizer': qtime, 'encoder': etime})
                    print(batch, frames, repeat, backend, 'quantizer', round(qtime['wall_ms_median'], 4),
                          'encoder', round(etime['wall_ms_median'], 4), 'exact',
                          all(c['exact'] for values in qchecks.values() for c in values) and torch.equal(ref.audio_codes, codes), flush=True)
        record['medians'] = {scope: {backend: statistics.median(r[scope]['wall_ms_median'] for r in record['rounds'] if r['backend'] == backend)
                                     for backend in ['none', 'triton']} for scope in ['quantizer', 'encoder']}
        record['all_exact'] = all(r['encoder_fidelity']['exact'] and all(c['exact'] for values in r['quantizer_fidelity'].values() for c in values)
                                  for r in record['rounds'])
        report['cases'].append(record)
        Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    report['all_exact'] = all(c['all_exact'] for c in report['cases'])
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['all_exact']: raise SystemExit('Quantizer fidelity failed')


if __name__ == '__main__': main()

"""Full-checkpoint and restored-context ablation of LFQ preparation fusion."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
from types import MethodType

import torch

from benchmarks.baseline import measure
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.quantizer_prepare import prepare, reference, compile_kernel
from benchmarks.residual_gemv_async_model import compare
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized


@contextmanager
def selected(model, enabled=True):
    import fast_moss.quantizer as quantizer
    original = quantizer.decode_latents
    saved = []
    counts = {'candidate_calls': 0, 'fallback_calls': 0, 'audit': False, 'preparation_checks': 0}

    def candidate(module, latents, *, straight_through=False):
        if (not enabled or latents.ndim != 3 or latents.shape[1] != 8
                or not latents.is_contiguous() or latents.dtype != torch.float32
                or not latents.is_cuda):
            counts['fallback_calls'] += 1
            return original(module, latents, straight_through=straight_through)
        counts['candidate_calls'] += 1
        enc, norm, twice, denom = prepare(latents)
        if counts['audit']:
            refs = reference(latents)
            for a, b in zip((enc, norm, twice, denom), refs):
                assert torch.equal(a.view(torch.int32), b.view(torch.int32)), 'Preparation bits differ'
                assert a.stride() == b.stride(), (a.stride(), b.stride())
                counts['preparation_checks'] += 1
        dots = twice @ module._fast_codebook.t()
        return quantizer.select(dots, norm, module._fast_codebook_norm,
                                module.codebook.weight, latents, straight_through=straight_through)

    try:
        quantizer.decode_latents = candidate
        for q in model.quantizer.quantizers:
            saved.append((q, q.decode_latents))
            q.decode_latents = MethodType(candidate, q)
        yield counts
    finally:
        torch.cuda.synchronize()
        quantizer.decode_latents = original
        for q, method in saved:
            q.decode_latents = method


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/full_quantizer_prepare.json')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--fidelity-only', action='store_true')
    p.add_argument('--timing-only', action='store_true')
    p.add_argument('--residual-backend', choices=['triton', 'cute'], default='triton')
    a = p.parse_args()
    if a.rounds < 1 or (a.fidelity_only and a.timing_only):
        p.error('Invalid benchmark options')
    model = load_model()
    clips, sources = audio_sources()
    opts = dict(options(), matrix_backend='cuda', norm_backend='cuda', ffn_backend='triton')
    opts['residual_backend'] = a.residual_backend
    report = {'scope': 'research LFQ preparation fusion; original FP32 full codec and paired supported runtime',
              'previous_commit': '9e9ec47', 'revision': REVISION, 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(), 'options': opts, 'sources': sources,
              'enabled_in_runtime': False, 'cases': [], 'timings': [],
              'timing_scope': f'{a.rounds} alternating-order rounds, 200 owned graph warmups, five samples of ten calls; copies and owned outputs included; loading/setup/capture excluded'}
    def save():
        Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    def run(x):
        enc = model._encode_frame(x)
        return enc.audio_codes, enc.encoder_hidden_states, model._decode_frame(enc.audio_codes).audio
    def inputs():
        yield from cases()
        for source, clip in zip(sources, clips):
            for b, t in [(2,1),(4,1),(8,1),(4,2),(1,2),(1,3),(8,3),(2,3),(1,6),(1,8),(1,12)]:
                idx = torch.arange(t*1920, device='cuda')[None]+torch.arange(b, device='cuda')[:,None]*1920
                yield f'{Path(source["path"]).stem}_b{b}_f{t}', clip[idx%clip.numel()][:,None], source
    for name, value, source in ([] if a.timing_only else inputs()):
        x = value.cuda()
        ref = run(x)
        with optimized(model, **opts), selected(model) as counts:
            counts['audit'] = True
            checks = {'eager': compare(ref, run(x))}
            counts['audit'] = False
            graph = GraphedCallable(run, x)
            checks['graph'] = compare(ref, graph(x))
            del graph
        checks['restored'] = compare(ref, run(x))
        exact = all(c['bits_equal'] for values in checks.values() for c in values)
        report['cases'].append({'name': name, 'shape': list(x.shape), 'source': source,
                                'checks': checks, **counts, 'all_exact': exact})
        save()
        print('fidelity', name, exact, counts, flush=True)
        if not exact:
            raise SystemExit('Full model fidelity failed')
    for batch, frames in ([] if a.fidelity_only else [(1,1),(8,1),(1,3),(8,3)]):
        idx = torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x = clips[1][idx%clips[1].numel()][:,None]
        ref = run(x)
        def encode(z):
            enc = model._encode_frame(z)
            return enc.audio_codes, enc.encoder_hidden_states
        methods = [('encode', encode, ref[:2]), ('roundtrip', run, ref)]
        record = {'batch': batch, 'frames': frames, 'rounds': []}
        for repeat in range(a.rounds):
            for mode in (['current','candidate'] if repeat%2 == 0 else ['candidate','current']):
                entry = {'round': repeat, 'mode': mode, 'results': {}}
                with optimized(model, **opts), selected(model, mode == 'candidate') as counts:
                    for direction, fn, refs in methods:
                        graph = GraphedCallable(fn, x)
                        checks = compare(refs, graph(x))
                        assert all(c['bits_equal'] for c in checks)
                        for _ in range(200):
                            graph(x)
                        torch.cuda.synchronize()
                        def many():
                            for _ in range(10):
                                graph(x)
                        timing = measure(many, warmup=2, repeats=5)
                        entry['results'][direction] = {'checks': checks, 'wall_ms': [v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry.update(counts)
                record['rounds'].append(entry)
                print('timing', batch, frames, repeat, mode,
                      {k: statistics.median(v['wall_ms']) for k,v in entry['results'].items()}, flush=True)
        record['medians_ms'] = {direction: {mode: statistics.median(v for r in record['rounds'] if r['mode']==mode
                                    for v in r['results'][direction]['wall_ms']) for mode in ['current','candidate']}
                                for direction,_,_ in methods}
        record['speedups'] = {k: v['current']/v['candidate'] for k,v in record['medians_ms'].items()}
        report['timings'].append(record)
        save()
    report['resources'] = compile_kernel()[2]
    report['all_exact'] = (all(c['all_exact'] for c in report['cases']) and
                          all(c['bits_equal'] for t in report['timings'] for r in t['rounds']
                              for v in r['results'].values() for c in v['checks']))
    report['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
    save()


if __name__ == '__main__':
    main()

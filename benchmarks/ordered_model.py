"""Research-only ordered Triton FFN replacement and full-checkpoint gates.

Requires the supported resident matrix context outside this context. All graphs
must be destroyed before context exit. No extra persistent weight copy is kept.
"""
from contextlib import contextmanager, nullcontext
import argparse
import json
from pathlib import Path
import statistics
from types import MethodType
import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_mm import tiled
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import load_model, REVISION
from fast_moss.matrices import folds_to_mm
from fast_moss.optimize import optimized


@contextmanager
def experimental(model):
    if not getattr(model, '_fast_matrix_runtime', None):
        raise ValueError('Enter supported matrix optimization before this research context')
    saved = []
    counts = {'calls': 0, 'warm_packs': 0}
    def wrap(original):
        def forward(module, x):
            if (x.dtype != torch.float32 or x.device != module.weight.device or x.requires_grad
                    or not x.is_contiguous() or not folds_to_mm(x)
                    or x.ndim < 2 or x.numel()//x.shape[-1] != 24
                    or x.shape[-1] != module.in_features or torch.is_autocast_enabled('cuda')):
                return original(x)
            if module.weight.is_contiguous():
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Warm the supported matrix before capturing')
                original(x)  # Let the supported owner pack and invalidate old graphs.
                counts['warm_packs'] += 1
            if module.weight.stride() != (1, module.out_features):
                raise RuntimeError('Expected supported resident KN storage')
            counts['calls'] += 1
            out = tiled(x.reshape(24, module.in_features), module.weight.T, 256, (32,128,32,4))
            return out.reshape(*x.shape[:-1], module.out_features)
        return forward
    try:
        for module in model.modules():
            if (type(module) is torch.nn.Linear and module.bias is None
                    and tuple(module.weight.shape) in {(5120,1280), (1280,5120)}):
                saved.append((module, module.forward))
                module.forward = MethodType(wrap(module.forward), module)
        yield counts
    finally:
        torch.cuda.synchronize()
        for module, original in reversed(saved):
            module.forward = original


def options():
    return dict(residual_backend='triton', rope_backend='triton', kv_backend='triton',
                share_rope_tables=True, attention_mask_backend='triton', quantizer_backend='triton',
                matrix_backend='cublaslt', projection_backend='triton')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--supported', action='store_true', help='Use the integrated matrix_backend=triton runtime')
    parser.add_argument('--output', default='results/full_ordered_model.json')
    args = parser.parse_args()
    model = load_model()
    clips, sources = audio_sources()
    report = {'scope': 'research-only exact ordered FP32 FFN; original full-checkpoint references',
              'revision': REVISION, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'dtype': 'float32', 'tf32': False, 'quantizers': 32, 'sources': sources,
              'shapes': [[24,5120,1280], [24,1280,5120]], 'chunk': 256, 'tile': [32,128,32,4],
              'options': options(), 'cases': [], 'timings': []}
    report['supported_runtime'] = args.supported
    if args.supported:
        report['scope'] = 'supported ordered FP32 FFN; original full-checkpoint references and paired backend timings'
    report['baseline_options'] = options()
    report['candidate_options'] = dict(options(), matrix_backend='triton') if args.supported else options()
    def run(x):
        enc = model._encode_frame(x)
        return enc.audio_codes, enc.encoder_hidden_states, model._decode_frame(enc.audio_codes).audio
    def inputs():
        yield from cases()
        for source, clip in zip(sources, clips):
            for b, t in [(8,3), (24,1), (1,24), (2,12)]:
                idx = torch.arange(t*1920, device='cuda')[None] + torch.arange(b, device='cuda')[:,None]*1920
                x = clip[idx % clip.numel()][:, None]
                yield f'{Path(source["path"]).stem}_b{b}_f{t}', x, source
    if not args.timing_only:
        for name, cpu, source in inputs():
            x = cpu.cuda()
            reference = run(x)
            r = {'name': name, 'shape': list(x.shape), 'source': source, 'comparisons': {}}
            opts = options()
            if args.supported:
                opts['matrix_backend'] = 'triton'
            with optimized(model, **opts), (nullcontext() if args.supported else experimental(model)) as counts:
                eager = run(x)
                graph = GraphedCallable(run, x)
                for mode, outputs in [('eager', eager), ('graph', graph(x))]:
                    r['comparisons'][mode] = {label: difference(ref, out) for label, ref, out in
                                              zip(['codes','hidden','audio'], reference, outputs)}
                r['counts'] = counts.copy() if counts else {'calls': model._fast_matrix_runtime.triton_calls}
                del graph, eager, outputs
            restored = run(x)
            r['comparisons']['restored'] = {label: difference(ref, out) for label, ref, out in
                                            zip(['codes','hidden','audio'], reference, restored)}
            r['all_exact'] = all(v['exact'] for mode in r['comparisons'].values() for v in mode.values())
            report['cases'].append(r)
            print(name, 'exact', r['all_exact'], r['counts'], flush=True)
            Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
            del reference, restored, x
    for b, t in [(8,3), (24,1), (1,24)]:
        idx = torch.arange(t*1920, device='cuda')[None] + torch.arange(b, device='cuda')[:,None]*1920
        x = clips[1][idx % clips[1].numel()][:, None]
        enc = model._encode_frame(x)
        codes = enc.audio_codes
        refs = {'encode': (codes, enc.encoder_hidden_states), 'decode': (model._decode_frame(codes).audio,)}
        rounds = []
        for repeat in range(3):
            for backend in (['cublaslt','triton'] if repeat%2 == 0 else ['triton','cublaslt']):
                opts = options()
                if args.supported and backend == 'triton':
                    opts['matrix_backend'] = 'triton'
                with optimized(model, **opts):
                    with experimental(model) if backend=='triton' and not args.supported else nullcontext() as counts:
                        r = {'round': repeat, 'backend': backend, 'results': {}}
                        def encode(z):
                            out = model._encode_frame(z)
                            return out.audio_codes, out.encoder_hidden_states
                        for name, fn, inp in [('encode', encode, x), ('decode', lambda z:(model._decode_frame(z).audio,), codes)]:
                            graph = GraphedCallable(fn, inp)
                            r['results'][name] = {'comparisons': [difference(ref,out) for ref,out in zip(refs[name],graph(inp))],
                                                  'timing': measure(lambda:graph(inp), repeats=20)}
                            del graph
                        r['counts'] = counts.copy() if counts else {'calls': model._fast_matrix_runtime.triton_calls}
                        rounds.append(r)
                        print(b, t, repeat, backend, {k:v['timing']['wall_ms_median'] for k,v in r['results'].items()}, flush=True)
        medians = {name:{backend: statistics.median(r['results'][name]['timing']['wall_ms_median'] for r in rounds if r['backend']==backend)
                         for backend in ['cublaslt','triton']} for name in refs}
        report['timings'].append({'batch':b,'frames':t,'rounds':rounds,'medians_ms':medians,
                                  'speedups': {name:v['cublaslt']/v['triton'] for name,v in medians.items()}})
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    report['all_exact'] = (all(r['all_exact'] for r in report['cases']) and
        all(c['exact'] for t in report['timings'] for r in t['rounds'] for v in r['results'].values() for c in v['comparisons']))
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['timing_scope'] = 'warmed graph input copies and owned outputs; excludes packing/capture/restoration'
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_exact']:
        raise SystemExit('Ordered matrix model fidelity failed')


if __name__ == '__main__':
    main()

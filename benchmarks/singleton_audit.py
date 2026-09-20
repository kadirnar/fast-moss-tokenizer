"""Isolate the B=24,T=1 full-checkpoint hidden-state regression."""
import argparse
import json
from pathlib import Path
import torch
from fast_moss.loading import load_model
from fast_moss.optimize import optimized
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.compare import difference


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--residuals', action='store_true', help='Compare same-input residual layouts and return native outputs')
    parser.add_argument('--output', default='results/ordered_singleton_audit.json')
    args = parser.parse_args()
    model = load_model()
    clips, sources = audio_sources()
    idx = torch.arange(1920,device='cuda')[None] + torch.arange(24,device='cuda')[:,None]*1920
    x = clips[1][idx % clips[1].numel()][:,None]
    ref = model._encode_frame(x).encoder_hidden_states
    if args.residuals:
        checks = []
        def wrap(name, fn):
            def run(x, update, scale):
                candidate, reference = fn(x, update, scale), x + update * scale
                checks.append({'layer':name, 'shape':list(x.shape), 'x_stride':list(x.stride()),
                               'update_stride':list(update.stride()), 'candidate_stride':list(candidate.stride()),
                               'reference_stride':list(reference.stride()), 'difference':difference(reference,candidate)})
                return reference
            return run
        with optimized(model, **options()):
            for name, module in model.named_modules():
                if hasattr(module, '_fast_scale_add'):
                    module._fast_scale_add = wrap(name, module._fast_scale_add)
            out = model._encode_frame(x).encoder_hidden_states
        report = {'scope':'same-input residual arithmetic/layout audit; returns native reference at each residual',
                  'checks':checks, 'hidden_after_reference_residuals':difference(ref,out)}
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        return
    configs = [('all',options()), ('defaults',{})]
    for key, value in options().items():
        opts = dict(options(), **{key: False if isinstance(value,bool) else 'none'})
        if key == 'rope_backend':
            opts['share_rope_tables'] = False
        configs.append((f'without_{key}', opts))
    report = {'scope': 'full-checkpoint singleton hidden-state isolation', 'sources':sources, 'records':[]}
    for label, opts in configs:
        with optimized(model, **opts):
            out = model._encode_frame(x).encoder_hidden_states
        r = {'label':label, 'options':opts, 'difference':difference(ref,out)}
        report['records'].append(r)
        print(r,flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Reuse supported streaming/profile gates with research LFQ preparation enabled."""
import argparse
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys

from benchmarks.quantizer_prepare_model import selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('gate', choices=['streaming', 'profile'])
    p.add_argument('--runtime', action='store_true')
    args, remaining = p.parse_known_args()
    if args.gate == 'streaming':
        import benchmarks.streaming_fidelity as gate
    else:
        import benchmarks.profile_graph as gate
    if '--output' not in remaining:
        p.error('Provide a distinct research --output report')
    output = Path(remaining[remaining.index('--output')+1])
    original = gate.optimized
    counts = []

    @contextmanager
    def research(model, *a, **kw):
        with original(model, *a, **kw), selected(model) as record:
            yield
        counts.append(record.copy())

    old_argv = sys.argv
    import benchmarks.quantizer_prepare_model as candidate
    old_runtime = candidate.RUNTIME
    try:
        candidate.RUNTIME = args.runtime
        gate.optimized = research
        sys.argv = [old_argv[0], *remaining]
        gate.main()
    finally:
        gate.optimized = original
        sys.argv = old_argv
        candidate.RUNTIME = old_runtime
    report = json.loads(output.read_text())
    if args.gate == 'profile':
        for direction, result in report['results'].items():
            trace = json.loads(gzip.open(output.with_suffix(f'.{direction}.json.gz'), 'rt').read())
            events = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel'
                      and e.get('name') == 'quantizer_prepare']
            result['kernel_summary']['quantizer_prepare_per_replay'] = len(events)/5
            result['kernel_summary']['quantizer_prepare_ms_per_replay'] = sum(e['dur'] for e in events)/5000
    report.update(quantizer_prepare_research=not args.runtime, enabled_in_runtime=args.runtime,
                  quantizer_prepare_counts=counts)
    output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

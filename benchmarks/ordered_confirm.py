"""Stress and paired warm/cold confirmation of ordered Triton FFN matrices."""
import json
from pathlib import Path
import statistics
import torch
import torch.nn.functional as F
from triton.testing import do_bench_cudagraph
from benchmarks.ordered_mm import tiled
from benchmarks.compare import difference
from benchmarks.matrices import evicted_replay
from fast_moss.cublaslt import LinearPlan
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import strict_precision, REVISION
from fast_moss.ordered_matrices import linear as runtime_linear


@torch.inference_mode()
def main():
    strict_precision()
    torch.manual_seed(4927)
    cases = torch.load('results/matrix_inputs.pt', weights_only=True)
    profile = json.loads(Path('fast_moss/matrix_profile.json').read_text())
    selected = {tuple(r['shape']): r for r in profile['records']}
    report = {'scope': 'actual-weight ordered FP32 GEMM confirmation; not model speedup',
              'revision': REVISION, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'tf32': False, 'seed': 4927, 'rounds': 3, 'records': []}
    flush = torch.empty(128*1024*1024, device='cuda', dtype=torch.uint8)
    for shape in [(24, 5120, 1280), (24, 1280, 5120)]:
        x, w = cases[shape]['x'], cases[shape]['weight']
        packed = w.T.contiguous()
        with LinearPlan(w, 24, 'packed', packed_weight=packed) as plan:
            index = plan.restore(selected[shape]['algorithm'], profile['cublaslt_version'])
            variants = [('actual', x), ('negated', -x), ('scaled', x*.17), ('zero', torch.zeros_like(x)),
                        ('subnormal', torch.randn_like(x)*1e-38), ('tiny', torch.randn_like(x)*1e-20),
                        ('large', torch.randn_like(x)*1e20)]
            variants += [(f'random_{i}', torch.randn_like(x)) for i in range(4)]
            sparse = torch.zeros_like(x); sparse[:, ::127] = torch.randn_like(sparse[:, ::127])
            variants.append(('sparse', sparse))
            fns = {'cublaslt': lambda z: plan(z, index)}
            fns['runtime_triton'] = lambda z: runtime_linear(z, packed)
            tiles = [(32,64,32,4), (32,128,32,4), (32,64,16,4), (32,128,16,4),
                     (32,128,32,8), (32,64,32,8), (16,128,32,8)]
            fns.update({str(tile): (lambda z, tile=tile: tiled(z, packed, 256, tile)) for tile in tiles})
            checks = []
            for name, fn in fns.items():
                graph = GraphedCallable(lambda z: (fn(z),), x)
                check = {'backend': name, 'variants': []}
                for label, z in variants:
                    ref = F.linear(z, w)
                    check['variants'].append({'name': label, 'eager': difference(ref, fn(z)),
                                               'graph': difference(ref, graph(z)[0])})
                del graph
                checks.append(check)
                print(shape, name, 'exact', all(v[m]['exact'] for v in check['variants'] for m in ['eager','graph']), flush=True)
            rounds = []
            for repeat in range(3):
                names = list(fns)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    fn = fns[name]
                    graph = GraphedCallable(lambda z: (fn(z),), x)
                    rounds.append({'round': repeat, 'backend': name,
                                   'warm_ms': do_bench_cudagraph(lambda: fn(x), rep=30),
                                   'cold': evicted_replay(graph, flush, repeats=25)})
                    del graph
            medians = {name: {'warm': statistics.median(r['warm_ms'] for r in rounds if r['backend']==name),
                              'cold': statistics.median(r['cold']['gpu_ms_median'] for r in rounds if r['backend']==name)}
                       for name in fns}
            r = {'shape': shape, 'checks': checks, 'timings': rounds, 'medians_ms': medians,
                 'all_exact': all(v[m]['exact'] for c in checks for v in c['variants'] for m in ['eager','graph'])}
            report['records'].append(r)
            print(shape, medians, flush=True)
            Path('results/ordered_confirm.json').write_text(json.dumps(report, indent=2)+'\n')
    report['all_exact'] = all(r['all_exact'] for r in report['records'])
    Path('results/ordered_confirm.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_exact']:
        raise SystemExit('Ordered matrix stress gate failed')


if __name__ == '__main__':
    main()

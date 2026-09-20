"""Independent synthetic preparation stress and fixed-graph component timing."""
import json
from pathlib import Path
import statistics

import torch

from benchmarks.quantizer_prepare import compile_kernel, prepare, reference
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    torch.manual_seed(1524)
    report = {'scope': 'research eight-channel LFQ preparation only; synthetic inputs; 32 preparations per fixed graph, five alternating rounds, nine samples of ten raw graph replays; excludes input copies, owned outputs, vendor dot, selection and full codec',
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'records': [], 'stress': []}
    for batch, time in [(1,4097),(257,17),(1025,1)]:
        for mode in ['random_bits', 'epsilon_boundary', 'unit']:
            if mode == 'random_bits':
                raw = torch.randint(0, 0x7f800000, (batch,8,time), device='cuda', dtype=torch.int32)
                sign = torch.randint(0,2,raw.shape,device='cuda',dtype=torch.int32) << 31
                x = (raw | sign).view(torch.float32)
            else:
                x = torch.randn(batch,8,time,device='cuda') * (3.535534e-13 if mode == 'epsilon_boundary' else 1.)
            refs = reference(x)
            graph = GraphedCallable(prepare, x)
            checks = []
            for name, out in [('eager', prepare(x)), ('graph', graph(x))]:
                for i,(a,b) in enumerate(zip(out,refs)):
                    exact = torch.equal(a.view(torch.int32), b.view(torch.int32))
                    checks.append({'mode':name, 'output':i, 'bits_equal':exact,
                                   'different_bits':int((a.view(torch.int32)!=b.view(torch.int32)).sum()),
                                   'elements':a.numel(), 'strides_equal':a.stride()==b.stride()})
            del graph
            report['stress'].append({'shape':[batch,8,time], 'input':mode, 'checks':checks})
            assert all(c['bits_equal'] and c['strides_equal'] for c in checks)
    for batch, time in [(1,1),(8,1),(1,3),(8,3),(1,40),(128,3),(8,40)]:
        x = torch.randn(batch,8,time,device='cuda')
        graphs = {name:GraphedCallable(lambda v, f=fn: tuple(out for _ in range(32) for out in f(v)),x)
                  for name,fn in [('current',reference),('candidate',prepare)]}
        record = {'shape':[batch,8,time], 'rounds':[]}
        for repeat in range(5):
            for name in (['current','candidate'] if repeat%2==0 else ['candidate','current']):
                graph = graphs[name]
                for _ in range(200):
                    graph.graph.replay()
                torch.cuda.synchronize()
                samples=[]
                for _ in range(9):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.graph.replay()
                    end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)/320)
                record['rounds'].append({'round':repeat, 'backend':name, 'samples_ms':samples})
        record['medians_ms'] = {name:statistics.median(v for r in record['rounds'] if r['backend']==name
                                                       for v in r['samples_ms']) for name in graphs}
        record['speedup'] = record['medians_ms']['current']/record['medians_ms']['candidate']
        report['records'].append(record)
        print(batch,time,record['medians_ms'],record['speedup'],flush=True)
        del graphs,graph
    report['resources'] = compile_kernel()[2]
    report['all_exact'] = all(c['bits_equal'] and c['strides_equal'] for r in report['stress'] for c in r['checks'])
    Path('results/quantizer_prepare_probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()

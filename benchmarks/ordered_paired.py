"""Interleaved model graphs sharing one resident matrix storage lifetime.

The runtime backend is changed only during construction; captured graphs have
fixed kernels. Warm every weight shape before either graph is captured.
"""
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


@torch.inference_mode()
def main():
    model = load_model()
    clips, sources = audio_sources()
    report = {'scope':'full checkpoint, 40 interleaved paired graph samples in one packed-weight lifetime',
              'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'cases':[],
              'timing_scope':'graph input copies and owned outputs; excludes load/packing/capture/restoration; two resident graphs per direction'}
    opts = options(); opts['matrix_backend'] = 'triton'
    for b, t in [(8,3),(24,1),(1,24)]:
        idx = torch.arange(t*1920,device='cuda')[None] + torch.arange(b,device='cuda')[:,None]*1920
        x = clips[1][idx % clips[1].numel()][:,None]
        enc = model._encode_frame(x)
        codes = enc.audio_codes
        refs = {'encode':(codes,enc.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        case = {'batch':b,'frames':t,'results':{}}
        with optimized(model, **opts):
            runtime = model._fast_matrix_runtime
            def encode(z):
                out = model._encode_frame(z)
                return out.audio_codes, out.encoder_hidden_states
            functions = [('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
            # Populate all weight storage before retaining any graph.
            for backend in ['cublaslt','triton']:
                runtime.backend = backend
                for _, fn, inp in functions:
                    fn(inp)
            for name, fn, inp in functions:
                graphs, checks = {}, {}
                for backend in ['cublaslt','triton']:
                    runtime.backend = backend
                    graphs[backend] = GraphedCallable(fn,inp)
                    checks[backend] = [difference(ref,out) for ref,out in zip(refs[name],graphs[backend](inp))]
                for _ in range(5):
                    for graph in graphs.values():
                        graph(inp)
                samples = []
                for repeat in range(40):
                    row = {'pair':repeat}
                    for backend in (['cublaslt','triton'] if repeat%2==0 else ['triton','cublaslt']):
                        row[backend] = measure(lambda:graphs[backend](inp),warmup=0,repeats=1)
                    samples.append(row)
                medians = {backend:statistics.median(r[backend]['wall_ms_median'] for r in samples)
                           for backend in graphs}
                differences = [r['cublaslt']['wall_ms_median']-r['triton']['wall_ms_median'] for r in samples]
                case['results'][name] = {'checks':checks,'samples':samples,'wall_ms_medians':medians,
                    'speedup':medians['cublaslt']/medians['triton'],'paired_saving_ms_median':statistics.median(differences),
                    'triton_faster_pairs':sum(d>0 for d in differences)}
                print(b,t,name,medians,'faster_pairs',sum(d>0 for d in differences),flush=True)
                del graph
                graphs.clear()
        report['cases'].append(case)
        Path('results/full_ordered_paired.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact'] = all(c['exact'] for case in report['cases'] for result in case['results'].values()
                             for checks in result['checks'].values() for c in checks)
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    Path('results/full_ordered_paired.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:
        raise SystemExit('Interleaved model gate failed')


if __name__ == '__main__':
    main()

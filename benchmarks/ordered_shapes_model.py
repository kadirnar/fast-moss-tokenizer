"""Expanded ordered matrices: original-reference fidelity and packing diagnostic.

Shared-storage timings are confounded; use ordered_shapes_ablation for speedups.
"""
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
import fast_moss.ordered_matrices as ordered

from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable


PREVIOUS_SHAPES = {(24,5120,1280),(24,1280,5120)}


@torch.inference_mode()
def main():
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton')
    report={'scope':'full checkpoint, expanded ordered matrix fidelity and 40 interleaved paired graph samples',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'options':opts,
            'timing_caveat':'Timing confound: a shared weight lifetime packs three previously native shapes, forcing extra contiguous weight copies in the previous graphs. These timings are diagnostic only, not accepted speedup evidence. Use full_ordered_shapes_ablation.json for independent restored lifetimes.',
            'matrix_configs':{str(k):v for k,v in ordered.CONFIGS.items()},'cases':[],'timings':[],
            'dispatch':'previous two ordered FFN shapes versus expanded attention/FFN matrices; both include the accepted FFN epilogues',
            'timing_scope':'same packed-weight lifetime, two graphs per direction; includes graph input copies and owned outputs; excludes loading, packing, capture and restoration'}
    def run(x):
        enc=model._encode_frame(x)
        return enc.audio_codes,enc.encoder_hidden_states,model._decode_frame(enc.audio_codes).audio
    def inputs():
        yield from cases()
        for source,clip in zip(sources,clips):
            for b,t in [(8,3),(24,1),(1,24),(2,12)]:
                idx=torch.arange(t*1920,device='cuda')[None]+torch.arange(b,device='cuda')[:,None]*1920
                yield f'{Path(source["path"]).stem}_b{b}_f{t}',clip[idx%clip.numel()][:,None],source
    for name,value,source in inputs():
        x=value.cuda();reference=run(x)
        r={'name':name,'shape':list(x.shape),'source':source,'comparisons':{}}
        with optimized(model,**opts):
            eager=run(x);graph=GraphedCallable(run,x)
            for mode,outputs in [('eager',eager),('graph',graph(x))]:
                r['comparisons'][mode]={label:difference(ref,out) for label,ref,out in zip(['codes','hidden','audio'],reference,outputs)}
            r['triton_calls']=model._fast_matrix_runtime.triton_calls
            r['ffn_calls']=model._fast_matrix_runtime.ffn_calls
            del graph,eager,outputs
        restored=run(x)
        r['comparisons']['restored']={label:difference(ref,out) for label,ref,out in zip(['codes','hidden','audio'],reference,restored)}
        r['all_exact']=all(v['exact'] for mode in r['comparisons'].values() for v in mode.values())
        report['cases'].append(r);print(name,r['all_exact'],r['ffn_calls'],flush=True)
        Path('results/full_ordered_shapes.json').write_text(json.dumps(report,indent=2)+'\n')
        del reference,restored,x
    for b,t in [(8,3),(24,1),(1,24)]:
        idx=torch.arange(t*1920,device='cuda')[None]+torch.arange(b,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None]
        enc=model._encode_frame(x);codes=enc.audio_codes
        refs={'encode':(codes,enc.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        case={'batch':b,'frames':t,'results':{}}
        with optimized(model,**opts):
            runtime=model._fast_matrix_runtime
            def encode(z):
                e=model._encode_frame(z)
                return e.audio_codes,e.encoder_hidden_states
            functions=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
            for enabled in [False,True]:
                ordered.SHAPES=set(ordered.CONFIGS) if enabled else PREVIOUS_SHAPES
                for _,fn,inp in functions:fn(inp)
            for name,fn,inp in functions:
                graphs={};checks={}
                for backend in ['previous','expanded']:
                    ordered.SHAPES=set(ordered.CONFIGS) if backend=='expanded' else PREVIOUS_SHAPES
                    graphs[backend]=GraphedCallable(fn,inp)
                    checks[backend]=[difference(ref,out) for ref,out in zip(refs[name],graphs[backend](inp))]
                for _ in range(5):
                    for graph in graphs.values():graph(inp)
                samples=[]
                for repeat in range(40):
                    row={'pair':repeat}
                    for backend in (['previous','expanded'] if repeat%2==0 else ['expanded','previous']):
                        row[backend]=measure(lambda:graphs[backend](inp),warmup=0,repeats=1)
                    samples.append(row)
                medians={backend:statistics.median(r[backend]['wall_ms_median'] for r in samples) for backend in graphs}
                savings=[r['previous']['wall_ms_median']-r['expanded']['wall_ms_median'] for r in samples]
                case['results'][name]={'checks':checks,'samples':samples,'wall_ms_medians':medians,
                    'speedup':medians['previous']/medians['expanded'],'paired_saving_ms_median':statistics.median(savings),
                    'expanded_faster_pairs':sum(v>0 for v in savings)}
                print(b,t,name,medians,'faster_pairs',sum(v>0 for v in savings),flush=True)
                del graph
                graphs.clear()
        report['timings'].append(case)
        Path('results/full_ordered_shapes.json').write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(r['all_exact'] for r in report['cases']) and all(
        c['exact'] for case in report['timings'] for result in case['results'].values() for checks in result['checks'].values() for c in checks)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path('results/full_ordered_shapes.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Expanded ordered matrix full-model gate failed')


if __name__=='__main__':
    try:
        main()
    finally:
        ordered.SHAPES=set(ordered.CONFIGS)

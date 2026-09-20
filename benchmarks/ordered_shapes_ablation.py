"""Compare expanded matrices with independent restored packing lifetimes.

A shared packed-weight lifetime penalizes old native fallback shapes with new
contiguous copies. This experiment creates each backend in its own context.
"""
import argparse
import json
from pathlib import Path
import statistics
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.ordered_shapes_model import PREVIOUS_SHAPES
from fast_moss.loading import load_model, REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
import fast_moss.ordered_matrices as ordered


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous-configs',help='JSON snapshot of prior shapes and unchanged configurations')
    parser.add_argument('--output',default='results/full_ordered_shapes_ablation.json')
    args=parser.parse_args()
    previous_shapes=PREVIOUS_SHAPES
    previous_commit=None
    if args.previous_configs:
        previous=json.loads(Path(args.previous_configs).read_text())
        previous_shapes={tuple(s) for s in previous['shapes']}
        previous_commit=previous['source_commit']
        if any(tuple(r['config'])!=ordered.CONFIGS.get(tuple(r['shape'])) for r in previous['configs']):
            raise ValueError('This ablation requires unchanged configs for the previous shapes')
    model=load_model();clips,sources=audio_sources()
    report={'scope':'full checkpoint expanded ordered matrices, independent packing lifetimes',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'dtype':'float32','tf32':False,'quantizers':32,'sources':sources,'cases':[],
            'timing_scope':'three alternating backend rounds, independent optimized contexts restored between backends; each context uses one live graph at a time, five samples of twenty owned graph calls per direction; includes input copies and outputs, excludes load/packing/capture/restoration',
            'previous_shapes':sorted(previous_shapes),'previous_commit':previous_commit,
            'expanded_configs':{str(k):v for k,v in ordered.CONFIGS.items()}}
    def encode(z):
        e=model._encode_frame(z)
        return e.audio_codes,e.encoder_hidden_states
    try:
        for b,t in [(8,3),(24,1),(1,24),(1,3)]:
            idx=torch.arange(t*1920,device='cuda')[None]+torch.arange(b,device='cuda')[:,None]*1920
            x=clips[1][idx%clips[1].numel()][:,None]
            enc=model._encode_frame(x);codes=enc.audio_codes
            refs={'encode':(codes,enc.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
            funcs=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
            r={'batch':b,'frames':t,'rounds':[]}
            for repeat in range(3):
                for backend in (['previous','expanded'] if repeat%2==0 else ['expanded','previous']):
                    ordered.SHAPES=set(ordered.CONFIGS) if backend=='expanded' else previous_shapes
                    entry={'round':repeat,'backend':backend,'results':{}}
                    with optimized(model,**dict(options(),matrix_backend='triton',ffn_backend='triton')):
                        runtime=model._fast_matrix_runtime
                        # Finish all packing before either managed graph exists.
                        for _,fn,inp in funcs:fn(inp)
                        entry['packed_bytes']=runtime.packed_bytes
                        entry['packed_weights']=len(runtime.packed)
                        entry['ordered_only_plans']=sum(p is None for p,_ in runtime.plans.values())
                        for name,fn,inp in funcs:
                            graph=GraphedCallable(fn,inp)
                            checks=[difference(ref,out) for ref,out in zip(refs[name],graph(inp))]
                            def run():
                                for _ in range(20):graph(inp)
                            timing=measure(run,warmup=2,repeats=5)
                            entry['results'][name]={'checks':checks,
                                'wall_ms_per_call':[v/20 for v in timing['wall_ms_samples']],
                                'gpu_ms_per_call':[v/20 for v in timing['gpu_ms_samples']]}
                            del graph
                    entry['restored']={name:[difference(ref,out) for ref,out in zip(refs[name],fn(inp))]
                                       for name,fn,inp in funcs}
                    r['rounds'].append(entry)
                    print(b,t,repeat,backend,{name:statistics.median(v['wall_ms_per_call'])
                          for name,v in entry['results'].items()},'packed',entry['packed_bytes'],flush=True)
            r['medians_ms']={name:{backend:statistics.median(v for entry in r['rounds'] if entry['backend']==backend
                                  for v in entry['results'][name]['wall_ms_per_call'])
                                  for backend in ['previous','expanded']} for name,_,_ in funcs}
            r['speedups']={name:times['previous']/times['expanded'] for name,times in r['medians_ms'].items()}
            r['all_exact']=all(c['exact'] for entry in r['rounds'] for result in entry['results'].values() for c in result['checks']) and all(
                c['exact'] for entry in r['rounds'] for checks in entry['restored'].values() for c in checks)
            report['cases'].append(r)
            Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    finally:
        ordered.SHAPES=set(ordered.CONFIGS)
    report['all_exact']=all(r['all_exact'] for r in report['cases'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Expanded matrix ablation gate failed')


if __name__=='__main__':main()

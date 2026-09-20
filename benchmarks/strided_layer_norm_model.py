"""Full-checkpoint fidelity and restored-context native LayerNorm ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.strided_layer_norm import layer_norm
from types import MethodType
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION, load_model
from fast_moss.optimize import optimized
RUNTIME = False


@contextmanager
def selected(model,configs):
    runtime=model._fast_norm_runtime
    counts={'candidate_calls':0}
    previous=runtime.strided_enabled
    runtime.strided_enabled=bool(configs) if RUNTIME else False
    if RUNTIME:
        before=runtime.strided_calls
        try:yield counts
        finally:
            counts['candidate_calls']=runtime.strided_calls-before
            runtime.strided_enabled=previous
        return
    saved=[]
    for module,original in list(runtime.forwards.items()):
        def forward(module,x,original=original):
            shape=tuple(x.shape)
            if (shape not in configs or x.ndim!=3 or x.stride()!=(shape[1]*shape[2],1,shape[1])
                    or x.dtype!=torch.float32 or x.device!=runtime.device or x.requires_grad
                    or torch.is_autocast_enabled('cuda') or module.eps!=1e-5
                    or module.weight is None or module.bias is None):return original(x)
            counts['candidate_calls']+=1
            return layer_norm(x,module.weight,module.bias,module.eps,configs[shape])[0]
        method=MethodType(forward,module)
        saved.append((module,original));runtime.forwards[module]=method;module.forward=method
    try:yield counts
    finally:
        try:torch.cuda.synchronize()
        finally:
            runtime.strided_enabled=previous
            for module,original in saved:
                runtime.forwards[module]=original;module.forward=original


def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    global RUNTIME
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_strided_layer_norm.json')
    args=parser.parse_args();RUNTIME=args.runtime
    confirmation=json.loads(Path('results/strided_layer_norm_confirm.json').read_text())
    configs={}
    for row in confirmation['records']:
        valid=[(name,t) for name,t in row['medians_ms'].items() if name!='native' and row['exact'][name]
               and t<row['medians_ms']['native']*.99]
        if valid:
            name,_=min(valid,key=lambda a:a[1]);configs[tuple(row['shape'])]=tuple(json.loads(name))
    if RUNTIME:
        from fast_moss.strided_normalization import CONFIGS
        configs=dict(CONFIGS)
    if not configs:raise SystemExit('No exact faster norm configurations')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':('supported runtime' if RUNTIME else 'research-only')+' strided CUDA LayerNorm, original checkpoint fidelity and paired current-runtime ablation',
        'previous_commit':'6bdd709','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'sources':sources,'options':opts,
        'configs':[{'shape':s,'config':c} for s,c in configs.items()],
        'timing_scope':'three rotating independently restored contexts; five samples of ten graph calls, input copies and owned outputs included; setup/capture/restoration excluded',
        'cases':[],'timings':[],'enabled_in_runtime':RUNTIME}
    output=Path(args.output)
    def save():output.write_text(json.dumps(report,indent=2)+'\n')
    def run(x):
        e=model._encode_frame(x)
        return e.audio_codes,e.encoder_hidden_states,model._decode_frame(e.audio_codes).audio
    def inputs():
        yield from cases()
        for source,clip in zip(sources,clips):
            for batch,frames in [(2,1),(4,1),(8,1),(4,2),(1,2),(1,3),(8,3),(2,3),(1,6),(1,8),(1,12)]:
                idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
                yield f'{Path(source["path"]).stem}_b{batch}_f{frames}',clip[idx%clip.numel()][:,None],source
    for name,value,source in inputs():
        x=value.cuda();ref=run(x)
        with optimized(model,**opts),selected(model,configs) as counts:
            eager=run(x);graph=GraphedCallable(run,x)
            checks={'eager':compare(ref,eager),'graph':compare(ref,graph(x))}
            del graph,eager
        checks['restored']=compare(ref,run(x))
        exact=all(c['bits_equal'] for values in checks.values() for c in values)
        report['cases'].append({'name':name,'shape':list(x.shape),'checks':checks,
                               **counts,'all_exact':exact})
        print('fidelity',name,exact,counts,flush=True);save()
        if not exact:raise SystemExit('Model fidelity mismatch')
        del ref,x
    for batch,frames in ([] if args.fidelity_only else [(1,1),(8,1),(1,3),(8,3)]):
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None];e=model._encode_frame(x);codes=e.audio_codes
        refs={'encode':(codes,e.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        def encode(z):
            e=model._encode_frame(z);return e.audio_codes,e.encoder_hidden_states
        fns=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
        record={'batch':batch,'frames':frames,'rounds':[]}
        for repeat in range(3):
            names=['current','candidate'] if repeat%2==0 else ['candidate','current']
            for backend in names:
                entry={'round':repeat,'backend':backend,'results':{}}
                with optimized(model,**opts),selected(model,configs if backend=='candidate' else {}) as counts:
                    for _,fn,inp in fns:fn(inp)
                    for name,fn,inp in fns:
                        graph=GraphedCallable(fn,inp)
                        checks=compare(refs[name],graph(inp))
                        if not all(c['bits_equal'] for c in checks):raise SystemExit('Timing fidelity mismatch')
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        entry['results'][name]={'checks':checks,'wall_ms':[v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry.update(counts);record['rounds'].append(entry)
                print('timing',batch,frames,repeat,backend,
                      {k:statistics.median(v['wall_ms']) for k,v in entry['results'].items()},flush=True)
        record['medians_ms']={name:{backend:statistics.median(v for r in record['rounds']
            if r['backend']==backend for v in r['results'][name]['wall_ms'])
            for backend in ('current','candidate')} for name,_,_ in fns}
        record['speedups']={name:t['current']/t['candidate'] for name,t in record['medians_ms'].items()}
        report['timings'].append(record);save()
    report['all_exact']=all(r['all_exact'] for r in report['cases'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()


if __name__=='__main__':main()

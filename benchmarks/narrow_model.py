"""Exact narrow-grid full-model gate and independent-context runtime ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
import torch
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.baseline import measure
from benchmarks.ordered_model import options
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
import fast_moss.small_matrices as small
import fast_moss.ordered_matrices as ordered

RUNTIME=False
BASELINE={}
ORDERED_BASELINE={}

@contextmanager
def active_tables(configs):
    previous=(small.CONFIGS,small.SHAPES,ordered.CONFIGS,ordered.SHAPES)
    previous_linear=small.linear
    layouts={shape:cfg[1:] for shape,cfg in configs.items() if cfg[0]=='layout'}
    if layouts:
        from benchmarks.small_layout import run as layout_run
        def research_linear(x,w):
            key=(x.shape[0],w.shape[0],x.shape[1])
            return layout_run(x,w,layouts[key]) if key in layouts else previous_linear(x,w)
        small.linear=research_linear
    small.CONFIGS={**BASELINE,**{s:c for s,c in configs.items() if c[0] not in ('ordered','layout')}}
    ordered.CONFIGS={**ORDERED_BASELINE,**{s:c[1:] for s,c in configs.items() if c[0]=='ordered'}}
    small.SHAPES=set(small.CONFIGS);ordered.SHAPES=set(ordered.CONFIGS)
    try:yield
    finally:
        small.CONFIGS,small.SHAPES,ordered.CONFIGS,ordered.SHAPES=previous
        small.linear=previous_linear


@torch.inference_mode()
def main():
    global RUNTIME,BASELINE,ORDERED_BASELINE
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--groups',action='store_true')
    parser.add_argument('--only-rows',type=int,nargs='+')
    parser.add_argument('--fidelity-only',action='store_true')
    parser.add_argument('--residual-backend',choices=['triton','cute'],default='triton')
    parser.add_argument('--output',default='results/full_narrow_rows.json')
    parser.add_argument('--selection',choices=['warm','cold'],default='cold')
    args=parser.parse_args();RUNTIME=args.runtime
    baseline=json.loads(Path('results/narrow_baseline.json').read_text())
    BASELINE={tuple(r['shape']):tuple(r['config']) for r in baseline['small_configs']}
    ORDERED_BASELINE={tuple(r['shape']):tuple(r['config']) for r in baseline['ordered_configs']}
    confirmation=json.loads(Path('results/narrow_confirm.json').read_text())
    configs={};cold_configs={}
    for record in confirmation['records']:
        prev=record['medians_ms']['previous']
        viable=[(name,t) for name,t in record['medians_ms'].items() if name not in ('native','previous')
                and t['warm']<prev['warm']*.99]
        if not viable:continue
        name,_=min(viable,key=lambda x:x[1]['warm'])
        configs[tuple(record['shape'])]=tuple(json.loads(name))
        cold=[(name,t) for name,t in viable if t['cold']<=prev['cold']*1.05]
        if cold:
            name,_=min(cold,key=lambda x:(x[1]['cold'],x[1]['warm']))
            cold_configs[tuple(record['shape'])]=tuple(json.loads(name))
    if args.only_rows:
        configs={s:c for s,c in configs.items() if s[0] in args.only_rows}
        cold_configs={s:c for s,c in cold_configs.items() if s[0] in args.only_rows}
    if args.groups and RUNTIME:raise ValueError('Group study is research-only')
    if RUNTIME:
        configs=cold_configs if args.selection=='cold' else configs
        if (small.CONFIGS!={**BASELINE,**{s:c for s,c in configs.items() if c[0] not in ('ordered','layout')}}
                or ordered.CONFIGS!={**ORDERED_BASELINE,**{s:c[1:] for s,c in configs.items() if c[0]=='ordered'}}):
            raise ValueError('Runtime table differs from selected candidates')
    groups={'previous':{},'candidate':configs}
    if not RUNTIME:groups['cold_subset']=cold_configs
    if args.groups:
        layout_confirmation=json.loads(Path('results/small_layout_confirm.json').read_text())
        layouts={}
        for row in layout_confirmation['records']:
            prev=row['medians_ms']['previous']
            valid=[(n,t) for n,t in row['medians_ms'].items() if n not in ('native','previous') and t['warm']<prev['warm']*.99]
            if valid:
                name,_=min(valid,key=lambda a:a[1]['warm'])
                layouts[tuple(row['shape'])]=('layout',*json.loads(name))
        one_two={s:c for s,c in configs.items() if s[0]<=2}
        groups={'previous':{},'one_two':one_two,'short':{s:c for s,c in configs.items() if s[0]<8},'layout':{**one_two,**layouts}}
        configs=groups['layout']
    backends=list(groups)
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton')
    opts['residual_backend']=args.residual_backend
    report={'scope':'exact narrow-grid matrices: full-model fidelity and independently restored ablation',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'configs':[{'shape':shape,'config':config} for shape,config in configs.items()],
            'cold_configs':[{'shape':shape,'config':config} for shape,config in cold_configs.items()],
            'options':opts,'sources':sources,'cases':[],'timings':[], 'runtime_dispatch':RUNTIME,
            'previous_commit':'fd559c3', 'baseline_scope':'preceding supported small/ordered tables and unchanged arithmetic', 'selection':args.selection if RUNTIME else 'group_study' if args.groups else 'warm_and_cold', 'only_rows':args.only_rows, 'group_configs':{name:[{'shape':s,'config':c} for s,c in cfg.items()] for name,cfg in groups.items()}, 'timing_scope':'three alternating independent-context rounds; five samples of ten graph calls, including owned outputs and input copies; excludes loading, capture and context setup'}
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
        with active_tables(configs),optimized(model,**opts):
            runtime=model._fast_matrix_runtime
            eager=run(x);graph=GraphedCallable(run,x)
            checks={mode:[difference(a,b) for a,b in zip(ref,out)]
                    for mode,out in [('eager',eager),('graph',graph(x))]}
            del graph,eager
        checks['restored']=[difference(a,b) for a,b in zip(ref,run(x))]
        record={'name':name,'shape':list(x.shape),'checks':checks,'candidate_calls':runtime.triton_calls,
                'all_exact':all(c['exact'] for values in checks.values() for c in values)}
        report['cases'].append(record)
        print('fidelity',name,record['all_exact'],runtime.triton_calls,flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        if not record['all_exact']:raise SystemExit('Full-model candidate mismatch')
        del ref,x
    for batch,frames in ([] if args.fidelity_only else [(1,1),(2,1),(8,1),(1,3),(8,3),(1,8)]):
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None];e=model._encode_frame(x);codes=e.audio_codes
        refs={'encode':(codes,e.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        def encode(z):
            e=model._encode_frame(z);return e.audio_codes,e.encoder_hidden_states
        fns=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
        record={'batch':batch,'frames':frames,'rounds':[]}
        for repeat in range(3):
            for backend in (backends[repeat%len(backends):]+backends[:repeat%len(backends)]):
                entry={'round':repeat,'backend':backend,'results':{}}
                active=groups[backend]
                with active_tables(active),optimized(model,**opts):
                    runtime=model._fast_matrix_runtime
                    for _,fn,inp in fns:fn(inp)
                    for name,fn,inp in fns:
                        graph=GraphedCallable(fn,inp)
                        checks=[difference(a,b) for a,b in zip(refs[name],graph(inp))]
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        entry['results'][name]={'checks':checks,'wall_ms':[v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry['calls']=runtime.triton_calls;record['rounds'].append(entry)
                print('timing',batch,frames,repeat,backend,{k:statistics.median(v['wall_ms']) for k,v in entry['results'].items()},flush=True)
        record['medians_ms']={name:{backend:statistics.median(v for r in record['rounds'] if r['backend']==backend
                                           for v in r['results'][name]['wall_ms']) for backend in backends}
                               for name,_,_ in fns}
        record['speedups']={name:{b:v['previous']/v[b] for b in backends if b!='previous'} for name,v in record['medians_ms'].items()}
        report['timings'].append(record)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases']) and all(c['exact'] for r in report['timings']
        for entry in r['rounds'] for out in entry['results'].values() for c in out['checks'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Model ablation mismatch')

if __name__=='__main__':main()

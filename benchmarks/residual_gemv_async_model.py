"""Full-checkpoint ablation of shared weight staging in residual GEMV."""
import argparse,json,statistics
from contextlib import contextmanager
from pathlib import Path
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.residual_gemv_async import linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION,load_model
from fast_moss.optimize import optimized
RUNTIME=False
@contextmanager
def selected(model,configs):
    """Temporarily replace only the one-row FFN contraction research path."""
    import fast_moss.ffn as ffn
    original=ffn.gemv_linear
    counts={'candidate_calls':0,'ffn_calls':0}
    runtime=model._fast_matrix_runtime;previous=runtime.residual_async_enabled
    if RUNTIME:
        from fast_moss.residual_async import CONFIG
        if configs and configs!={5120:CONFIG}:raise ValueError('Runtime selection differs from validated schedule')
        before=runtime.residual_async_calls;runtime.residual_async_enabled=bool(configs)
        try:yield counts
        finally:
            counts.update(candidate_calls=runtime.residual_async_calls-before,ffn_calls=runtime.residual_async_calls-before)
            runtime.residual_async_enabled=previous
        return
    runtime.residual_async_enabled=False
    def candidate(x,w,mode,residual,scale,library):
        if mode!='residual' or tuple(x.shape)!=(1,5120) or tuple(w.shape)!=(1280,5120) or 5120 not in configs:
            return original(x,w,mode,residual,scale,library)
        counts['candidate_calls']+=1;counts['ffn_calls']+=1
        return linear(x,w,residual,scale,configs[5120])
    ffn.gemv_linear=candidate
    try:yield counts
    finally:
        torch.cuda.synchronize();ffn.gemv_linear=original;runtime.residual_async_enabled=previous


def runtime_count(model,counts):
    return model._fast_matrix_runtime.residual_async_calls if RUNTIME else counts['candidate_calls']

def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    global RUNTIME
    parser=argparse.ArgumentParser()
    parser.add_argument("--runtime",action="store_true")
    parser.add_argument('--selection',default='results/residual_gemv_async_confirm.json')

    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--extra-warmup-replays',type=int,default=0)
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_residual_gemv_async.json')
    args=parser.parse_args();RUNTIME=args.runtime
    if args.rounds<1 or args.extra_warmup_replays<0 or (args.timing_only and args.fidelity_only):parser.error('Invalid timing options')
    selection=json.loads(Path(args.selection).read_text());configs={}
    for row in selection['records']:
        ratios=row['rings'][-1]['speedups'];best=max(ratios,key=ratios.get)
        if row['shape'][2]==5120 and ratios[best]>1.005:configs[5120]=tuple(json.loads(best))
    if not configs:raise SystemExit('No faster confirmed configuration')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':('supported' if RUNTIME else 'research')+' asynchronous FFN residual GEMV weight staging, full checkpoint and paired current-runtime ablation','candidate_backend':'cuda',
        'previous_commit':'b2a1086','integration_parent':'c4f560a','selection_report':args.selection,'revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'sources':sources,'options':opts,
        'configs':configs,
        'fidelity_skipped':args.timing_only,'timing_rounds':args.rounds,
        'timing_scope':'rotating independently restored contexts; five samples of ten graph calls, input copies and owned outputs included; setup/capture/restoration excluded',
        'cases':[],'timings':[],'enabled_in_runtime':RUNTIME,'extra_warmup_replays':args.extra_warmup_replays}
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
    for name,value,source in ([] if args.timing_only else inputs()):
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
        for repeat in range(args.rounds):
            names=['current','candidate'] if repeat%2==0 else ['candidate','current']
            for backend in names:
                entry={'round':repeat,'backend':backend,'results':{}}
                with optimized(model,**opts),selected(model,configs if backend=='candidate' else {}) as counts:
                    for _,fn,inp in fns:fn(inp)
                    for name,fn,inp in fns:
                        before=runtime_count(model,counts)
                        graph=GraphedCallable(fn,inp)
                        checks=compare(refs[name],graph(inp))
                        if not all(c['bits_equal'] for c in checks):raise SystemExit('Timing fidelity mismatch')
                        for _ in range(args.extra_warmup_replays):graph(inp)
                        torch.cuda.synchronize()
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        after=runtime_count(model,counts)
                        entry['results'][name]={'checks':checks,'candidate_capture_calls':after-before,'wall_ms':[v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry.update(counts);record['rounds'].append(entry)
                print('timing',batch,frames,repeat,backend,
                      {k:statistics.median(v['wall_ms']) for k,v in entry['results'].items()},flush=True)
        record['medians_ms']={name:{backend:statistics.median(v for r in record['rounds']
            if r['backend']==backend for v in r['results'][name]['wall_ms'])
            for backend in ('current','candidate')} for name,_,_ in fns}
        record['speedups']={name:t['current']/t['candidate'] for name,t in record['medians_ms'].items()}
        report['timings'].append(record);save()
    report['all_exact']=all(r['all_exact'] for r in report['cases']) and all(c['bits_equal'] for t in report['timings'] for r in t['rounds'] for v in r['results'].values() for c in v['checks'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()


if __name__=='__main__':main()

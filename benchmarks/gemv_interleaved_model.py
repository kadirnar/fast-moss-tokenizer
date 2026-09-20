"""Full-checkpoint fidelity and restored-context interleaved GEMV candidate ablation."""
import argparse
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.gemv_interleaved import pack,gemv
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION, load_model
from fast_moss.optimize import optimized
import fast_moss.small_matrices as small


@contextmanager
def selected(configs):
    original=small.linear
    counts={'candidate_calls':0,'pack_count':0,'packed_bytes':0,'packing_wall_ms':0.}
    cache={}
    def linear(x,w):
        shape=(x.shape[0],w.shape[0],x.shape[1])
        if shape in configs:
            lanes,group,warps,unroll=configs[shape]
            key=(w.data_ptr(),w._version,tuple(w.shape),lanes,group)
            if key not in cache:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Warm interleaved weights before graph capture')
                torch.cuda.synchronize();start=time.perf_counter()
                p=pack(w,lanes,group)
                torch.cuda.synchronize()
                counts['packing_wall_ms']+=(time.perf_counter()-start)*1000
                counts['packed_bytes']+=p.data.numel()*p.data.element_size()
                counts['pack_count']+=1
                cache[key]=(w,p)
            counts['candidate_calls']+=1
            return gemv(x,cache[key][1],warps,unroll)
        return original(x,w)
    small.linear=linear
    try:yield counts
    finally:
        try:torch.cuda.synchronize()
        finally:
            small.linear=original
            cache.clear()


def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection',choices=['warm','cold'],default='warm')
    parser.add_argument('--output',default='results/full_gemv_interleaved.json')
    args=parser.parse_args()
    confirmation=json.loads(Path('results/gemv_interleaved_confirm.json').read_text())
    configs={}
    for row in confirmation['records']:
        candidates=[(name,timing) for name,timing in row['medians_ms'].items()
            if name not in ('native','current') and row['exact'][name]
            and timing['warm']<row['medians_ms']['current']['warm']*.99]
        if args.selection=='cold':
            candidates=[(name,t) for name,t in candidates if t['cold']<=row['medians_ms']['current']['cold']*1.05]
        if candidates:
            name,_=min(candidates,key=lambda t:(t[1]['cold'],t[1]['warm']) if args.selection=='cold' else t[1]['warm'])
            configs[tuple(row['shape'])]=tuple(json.loads(name))
    if not configs:raise SystemExit('No confirmed warm-cache candidate')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton')
    report={'scope':'research-only interleaved FP32 GEMV storage, original checkpoint fidelity and paired current-runtime ablation',
        'previous_commit':'dafaa46','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'sources':sources,'options':opts,
        'configs':[{'shape':s,'config':c} for s,c in configs.items()],
        'timing_scope':'three rotating independently restored contexts; five samples of ten graph calls, input copies and owned outputs included; setup/capture/restoration excluded',
        'cases':[],'timings':[],'enabled_in_runtime':False,'selection':args.selection}
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
        with selected(configs) as counts,optimized(model,**opts):
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
    for batch,frames in [(1,1),(8,1),(1,3)]:
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
                with selected(configs if backend=='candidate' else {}) as counts,optimized(model,**opts):
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

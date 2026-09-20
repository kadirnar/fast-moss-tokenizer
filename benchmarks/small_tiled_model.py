"""Research-only full-model gate and independent-context small-matrix ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from benchmarks.small_tiled_grouped import grouped
from benchmarks.small_tiled_split import split
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.baseline import measure
from benchmarks.ordered_model import options
from fast_moss.loading import load_model,REVISION
from fast_moss.optimize import optimized
from fast_moss.graphs import GraphedCallable
from fast_moss.matrices import folds_to_mm
import fast_moss.small_matrices as small

RUNTIME = False


@contextmanager
def candidate(model,configs):
    previous=small.SHAPES
    small.SHAPES=set(configs) if RUNTIME else set()
    if RUNTIME:
        runtime=model._fast_matrix_runtime
        class Calls:
            def __getitem__(self,index):return runtime.small_calls
        try:yield Calls()
        finally:small.SHAPES=previous
        return
    saved=[];calls=[0]
    for module in model.modules():
        if type(module) is not torch.nn.Linear or module.bias is not None:continue
        original=module.forward;owned='forward' in module.__dict__
        def forward(module,x,*args,_original=original,**kwargs):
            shape=(x.numel()//x.shape[-1],*module.weight.shape) if x.ndim else None
            if (shape in configs and not args and not kwargs and x.is_contiguous() and folds_to_mm(x)
                    and module.weight.is_contiguous() and x.dtype==torch.float32):
                name,config=configs[shape];calls[0]+=1
                y={'grouped':grouped,'split':split}[name](x.reshape(shape[0],shape[-1]),module.weight,*config)
                return y.reshape(*x.shape[:-1],module.out_features)
            return _original(x,*args,**kwargs)
        saved.append((module,original,owned));module.forward=MethodType(forward,module)
    try:yield calls
    finally:
        small.SHAPES=previous
        for module,original,owned in reversed(saved):
            if owned:module.forward=original
            else:del module.forward


@torch.inference_mode()
def main():
    global RUNTIME
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--fidelity-only',action='store_true')
    parser.add_argument('--residual-backend',choices=['triton','cute'],default='triton')
    parser.add_argument('--output',default='results/full_small_tiled.json')
    args=parser.parse_args();RUNTIME=args.runtime
    confirmation=json.loads(Path('results/small_tiled_confirm.json').read_text())
    configs={}
    for record in confirmation['records']:
        native=record['medians_ms']['native']
        candidates=[(name,t) for name,t in record['medians_ms'].items() if name!='native'
                    and t['warm']<native['warm']*.98 and t['cold']<native['cold']*.98]
        if candidates:
            best=min(candidates,key=lambda x:x[1]['cold'])[0]
            configs[tuple(record['shape'])]=next((name,config) for name,config in record['configs']
                                               if f'{name}_{config}'==best)
    if not configs:raise SystemExit('No candidate passed both component timing gates')
    if RUNTIME and any((name,*config,*([16] if name=='split' else [])) != small.CONFIGS[shape]
                       for shape,(name,config) in configs.items()):
        raise ValueError('Runtime configurations differ from confirmed candidates')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='triton',ffn_backend='triton')
    opts['residual_backend']=args.residual_backend
    report={'scope':'research-only tiled small matrices: full model original-reference fidelity and independent-context ablation',
            'revision':REVISION,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'configs':[{'shape':shape,'backend':name,'config':config} for shape,(name,config) in configs.items()],
            'options':opts,'sources':sources,'cases':[],'timings':[], 'runtime_dispatch':RUNTIME,
            'previous_commit':'ba4efdd', 'timing_scope':'three alternating independent-context rounds; five samples of ten graph calls, including owned outputs and input copies; excludes loading, capture and context setup'}
    def run(x):
        e=model._encode_frame(x)
        return e.audio_codes,e.encoder_hidden_states,model._decode_frame(e.audio_codes).audio
    def inputs():
        yield from cases()
        for source,clip in zip(sources,clips):
            for batch,frames in [(1,3),(2,3),(1,6),(1,12)]:
                idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
                yield f'{Path(source["path"]).stem}_b{batch}_f{frames}',clip[idx%clip.numel()][:,None],source
    for name,value,source in inputs():
        x=value.cuda();ref=run(x)
        with optimized(model,**opts),candidate(model,configs) as calls:
            eager=run(x);graph=GraphedCallable(run,x)
            checks={mode:[difference(a,b) for a,b in zip(ref,out)]
                    for mode,out in [('eager',eager),('graph',graph(x))]}
            del graph,eager
        checks['restored']=[difference(a,b) for a,b in zip(ref,run(x))]
        record={'name':name,'shape':list(x.shape),'checks':checks,'candidate_calls':calls[0],
                'all_exact':all(c['exact'] for values in checks.values() for c in values)}
        report['cases'].append(record)
        print('fidelity',name,record['all_exact'],calls[0],flush=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
        if not record['all_exact']:raise SystemExit('Full-model candidate mismatch')
        del ref,x
    for batch,frames in ([] if args.fidelity_only else [(1,3),(1,1),(2,3),(1,6)]):
        idx=torch.arange(frames*1920,device='cuda')[None]+torch.arange(batch,device='cuda')[:,None]*1920
        x=clips[1][idx%clips[1].numel()][:,None];e=model._encode_frame(x);codes=e.audio_codes
        refs={'encode':(codes,e.encoder_hidden_states),'decode':(model._decode_frame(codes).audio,)}
        def encode(z):
            e=model._encode_frame(z);return e.audio_codes,e.encoder_hidden_states
        fns=[('encode',encode,x),('decode',lambda z:(model._decode_frame(z).audio,),codes)]
        record={'batch':batch,'frames':frames,'rounds':[]}
        for repeat in range(3):
            for backend in (['previous','candidate'] if repeat%2==0 else ['candidate','previous']):
                entry={'round':repeat,'backend':backend,'results':{}}
                with optimized(model,**opts),candidate(model,configs if backend=='candidate' else {}) as calls:
                    for _,fn,inp in fns:fn(inp)
                    for name,fn,inp in fns:
                        graph=GraphedCallable(fn,inp)
                        checks=[difference(a,b) for a,b in zip(refs[name],graph(inp))]
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        entry['results'][name]={'checks':checks,'wall_ms':[v/10 for v in timing['wall_ms_samples']]}
                        del graph
                entry['calls']=calls[0];record['rounds'].append(entry)
                print('timing',batch,frames,repeat,backend,{k:statistics.median(v['wall_ms']) for k,v in entry['results'].items()},flush=True)
        record['medians_ms']={name:{backend:statistics.median(v for r in record['rounds'] if r['backend']==backend
                                           for v in r['results'][name]['wall_ms']) for backend in ['previous','candidate']}
                               for name,_,_ in fns}
        record['speedups']={name:v['previous']/v['candidate'] for name,v in record['medians_ms'].items()}
        report['timings'].append(record)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    report['all_exact']=all(c['all_exact'] for c in report['cases']) and all(c['exact'] for r in report['timings']
        for entry in r['rounds'] for out in entry['results'].values() for c in out['checks'])
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    if not report['all_exact']:raise SystemExit('Model ablation mismatch')

if __name__=='__main__':main()

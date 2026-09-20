"""Research full-checkpoint attention projection/residual fusion ablation."""
import argparse,json,statistics
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from benchmarks.attention_residual import CONFIGS,linear
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION,load_model
from fast_moss.optimize import optimized

@contextmanager
def selected(model,configs):
    counts={'candidate_calls':0,'contiguous_calls':0,'strided_calls':0,'shapes':{}};saved=[]
    def replace(obj,name,value):
        saved.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    try:
        if configs:
            for layer in model.modules():
                if type(layer).__name__!='MossAudioTokenizerTransformerLayer':continue
                if len(layer.self_attn.out_projs)!=1:continue
                projection=layer.self_attn.out_projs[0]
                if projection not in model._fast_matrix_runtime.forwards:continue
                pending=[];original_sa=layer._sa_block;original_proj=projection.forward;original_scale=layer._fast_scale_add
                def project(self,x,original=original_proj,pending=pending,layer=layer,**kwargs):
                    if pending and not kwargs:
                        state=pending[-1];residual=state['residual'];shape=(x.numel()//x.shape[-1],*self.weight.shape)
                        if (json.dumps(shape) in configs and x.is_contiguous() and self.weight.is_contiguous()
                            and x.dtype==torch.float32 and not x.requires_grad):
                            out=linear(x.reshape(shape[0],shape[-1]),self.weight,residual,layer.layer_scale_1.scale)
                            state['fused']=True;counts['candidate_calls']+=1
                            counts['contiguous_calls' if residual.is_contiguous() else 'strided_calls']+=1
                            key=json.dumps(shape);counts['shapes'][key]=counts['shapes'].get(key,0)+1
                            return out
                    return original(x,**kwargs)
                def scale(x,update,s,original=original_scale,pending=pending):
                    return update if pending and pending[-1]['fused'] and x is pending[-1]['residual'] else original(x,update,s)
                def sa(self,x,original=original_sa,pending=pending,projection=projection):
                    modules=(self.norm1,self.self_attn,self.self_attn.in_projs[0],projection,self.layer_scale_1)
                    if (x.ndim!=3 or x.dtype!=torch.float32 or x.requires_grad or torch.is_autocast_enabled('cuda')
                        or (not x.is_contiguous() and x.stride()!=(x.shape[1]*x.shape[2],1,x.shape[1]))
                        or torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks
                        or any(m._forward_hooks or m._forward_pre_hooks for m in modules)):return original(x)
                    pending.append({'residual':x,'fused':False})
                    try:return original(x)
                    finally:pending.pop()
                replace(projection,'forward',MethodType(project,projection));replace(layer,'_fast_scale_add',scale);replace(layer,'_sa_block',MethodType(sa,layer))
        yield counts
    finally:
        torch.cuda.synchronize()
        for obj,name,original in reversed(saved):setattr(obj,name,original)

def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['all','one','multi'],default='all')

    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--extra-warmup-replays',type=int,default=0)
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_attention_residual.json')
    args=parser.parse_args()
    if args.rounds<1 or args.extra_warmup_replays<0 or (args.timing_only and args.fidelity_only):parser.error('Invalid timing options')
    configs={json.dumps(shape):cfg for shape,cfg in CONFIGS.items() if args.group=='all' or (shape[0]==1)==(args.group=='one')}
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':'research attention output projection/residual fusion'+', full checkpoint and paired current-runtime ablation','candidate_backend':'triton+cuda',
        'group':args.group,'previous_commit':'84caafc','revision':REVISION,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(),'sources':sources,'options':opts,
        'configs':configs,
        'fidelity_skipped':args.timing_only,'timing_rounds':args.rounds,
        'timing_scope':'rotating independently restored contexts; five samples of ten graph calls, input copies and owned outputs included; setup/capture/restoration excluded',
        'cases':[],'timings':[],'enabled_in_runtime':False,'extra_warmup_replays':args.extra_warmup_replays}
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
                        before=counts['candidate_calls']
                        graph=GraphedCallable(fn,inp)
                        checks=compare(refs[name],graph(inp))
                        if not all(c['bits_equal'] for c in checks):raise SystemExit('Timing fidelity mismatch')
                        for _ in range(args.extra_warmup_replays):graph(inp)
                        torch.cuda.synchronize()
                        def run_many():
                            for _ in range(10):graph(inp)
                        timing=measure(run_many,warmup=2,repeats=5)
                        after=counts['candidate_calls']
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

"""Full-checkpoint fidelity and restored-context LayerNorm/projection fusion ablation."""
import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from benchmarks.baseline import measure
from benchmarks.compare import difference
from benchmarks.fidelity import cases
from fast_moss.ffn import math_library,observed,owned_norm_forward
import torch.nn.functional as F
from types import MethodType
from benchmarks.lane_completion import audio_sources
from benchmarks.ordered_model import options
from fast_moss.graphs import GraphedCallable
from fast_moss.loading import REVISION, load_model
from fast_moss.optimize import optimized


@contextmanager
def selected(model,configs):
    from benchmarks.norm_gemv import linear as fused
    from fast_moss.normalization import owned_forward
    runtime=model._fast_matrix_runtime;counts={'candidate_calls':0,'qkv_calls':0,'ffn_calls':0};saved=[]
    owned={id(m) for m,_ in runtime.saved};library=math_library()
    def set_method(module,name,fn):
        saved.append((module,name,getattr(module,name)));setattr(module,name,MethodType(fn,module))
    def eligible(x,norm,weight):
        return (x.ndim==3 and tuple(x.shape)==(1,1,1280) and x.is_contiguous() and x.dtype==torch.float32
            and not x.requires_grad and not torch.is_autocast_enabled('cuda')
            and owned_forward(norm) and norm.eps==1e-5 and weight.is_contiguous())
    for layer in model.modules():
        if type(layer).__name__!='MossAudioTokenizerTransformerLayer':continue
        if ('gelu' in configs and id(layer.linear1) in owned and id(layer.linear2) in owned
            and tuple(layer.linear1.weight.shape)==(5120,1280)):
            original=layer._ff_block
            def ff(self,x,original=original):
                if (observed(self) or self.activation is not F.gelu or self.gating is not None or self.weights_per_step
                    or not eligible(x,self.norm2,self.linear1.weight)):return original(x)
                hidden=fused(x.reshape(1,1280),self.linear1.weight,self.norm2.weight,self.norm2.bias,self.norm2.eps,'gelu',configs['gelu'])
                counts['candidate_calls']+=1;counts['ffn_calls']+=1
                return self.linear2(hidden.reshape(1,1,5120),_fast_epilogue=('residual',x,self.layer_scale_2.scale,library))
            set_method(layer,'_ff_block',ff)
        if ('none' in configs and len(layer.self_attn.in_projs)==1
            and id(layer.self_attn.in_projs[0]) in owned
            and tuple(layer.self_attn.in_projs[0].weight.shape)==(3840,1280)):
            norm=layer.norm1;projection=layer.self_attn.in_projs[0];pending=[]
            original_sa=layer._sa_block;original_norm=norm.forward;original_proj=projection.forward
            def norm_forward(self,x,original=original_norm,pending=pending):
                return x if pending and x is pending[0] else original(x)
            def projection_forward(self,x,original=original_proj,pending=pending,norm=norm):
                if not pending or x is not pending[0]:return original(x)
                y=fused(x.reshape(1,1280),self.weight,norm.weight,norm.bias,norm.eps,'none',configs['none'])
                counts['candidate_calls']+=1;counts['qkv_calls']+=1
                return y.reshape(1,1,3840)
            # The norm forward is temporarily intercepted only within this SA call.
            def sa_checked(self,x,original=original_sa,norm=norm,projection=projection,pending=pending):
                modules=(self.self_attn,norm,projection,self.layer_scale_1)
                if (tuple(x.shape)!=(1,1,1280) or not x.is_contiguous() or x.dtype!=torch.float32 or x.requires_grad
                    or torch.is_autocast_enabled('cuda') or norm.eps!=1e-5 or not projection.weight.is_contiguous()
                    or torch.nn.modules.module._global_forward_hooks or torch.nn.modules.module._global_forward_pre_hooks
                    or any(m._forward_hooks or m._forward_pre_hooks for m in modules)):return original(x)
                pending.append(x)
                try:return original(x)
                finally:pending.clear()
            if not owned_forward(norm):continue
            set_method(layer,'_sa_block',sa_checked);set_method(norm,'forward',norm_forward);set_method(projection,'forward',projection_forward)
    try:yield counts
    finally:
        torch.cuda.synchronize()
        for module,name,original in reversed(saved):setattr(module,name,original)


def compare(ref,out):
    return [{**difference(a,b),'bits_equal':torch.equal(
        a.view(torch.int32) if a.dtype==torch.float32 else a,
        b.view(torch.int32) if b.dtype==torch.float32 else b)} for a,b in zip(ref,out)]


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--confirmation',default='results/norm_gemv_confirm.json')

    parser.add_argument('--fidelity-only', action='store_true')
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--extra-warmup-replays',type=int,default=0)
    parser.add_argument('--residual-backend', choices=['triton','cute'], default='triton')
    parser.add_argument('--output', default='results/full_norm_gemv.json')
    args=parser.parse_args()
    if args.rounds<1 or args.extra_warmup_replays<0 or (args.timing_only and args.fidelity_only):parser.error('Invalid timing options')
    confirmation=json.loads(Path(args.confirmation).read_text())
    configs={}
    for r in confirmation['records']:
        speeds=r['rings'][-1]['speedups'];name=max(speeds,key=speeds.get)
        if speeds[name]>1.005:configs[r['mode']]=tuple(json.loads(name))
    if not configs:raise SystemExit('No exact faster selected configurations')
    model=load_model();clips,sources=audio_sources()
    opts=dict(options(),matrix_backend='cuda',ffn_backend='triton',norm_backend='cuda');opts['residual_backend']=args.residual_backend
    report={'scope':'research LayerNorm/projection fusions'+', full checkpoint and paired current-runtime ablation','candidate_backend':'cuda',
        'selection_report':args.confirmation,'previous_commit':'b240bd2','revision':REVISION,'torch':torch.__version__,
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
    for batch,frames in ([] if args.fidelity_only else [(1,1),(8,1),(1,3)]):
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
